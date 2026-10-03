from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import torch

from .songnet import SongNetPaper


_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "songnet_paper.pt"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SAMPLE_RATE = 22050
_DURATION = 30
_GENRES = (
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
)


def _preprocess_audio(path):
    try:
        y, sr = librosa.load(path, sr=_SAMPLE_RATE, mono=True, duration=_DURATION)
    except Exception as exc:
        raise ValueError(f"Cannot read audio file: {path}") from exc
    target_samples = _SAMPLE_RATE * _DURATION
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


@lru_cache(maxsize=1)
def _load_model():
    if not _CHECKPOINT.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {_CHECKPOINT}")
    state = torch.load(_CHECKPOINT, map_location=_DEVICE, weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = SongNetPaper(n_mels=128, num_classes=8, dropout=0.4).to(_DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_genre(audio_path):
    """Return the predicted genre and probabilities for all eight genres."""
    if audio_path is None:
        raise ValueError("Please select an audio file.")
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    model = _load_model()
    mel = _preprocess_audio(path)
    inputs = torch.from_numpy(mel).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0].cpu().tolist()
    scores = dict(zip(_GENRES, probabilities))
    return max(scores, key=scores.get), scores
