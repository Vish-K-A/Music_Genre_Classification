from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

EXPECTED_GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]
EXPECTED_TRACK_COUNT = 8000
EXPECTED_FEATURE_COUNT = 140
FEATURE_NAME = "mfcc"
GENRE_ORDER = EXPECTED_GENRES

__all__ = [
    "GENRE_ORDER",
    "EXPECTED_GENRES",
    "FEATURE_NAME",
    "load_fma_small_data",
    "split_and_scale",
    "get_baseline_models",
    "train_baselines",
    "evaluate_baselines",
    "predict_genre",
    "save_baseline_bundle",
    "load_baseline_bundle",
]


def _resolve_data_dir(data_dir: str | Path) -> Path:
    """Resolve the FMA metadata directory from a provided path or the repo layout."""
    candidate = Path(data_dir).expanduser()
    if candidate.exists():
        return candidate

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    for path in [
        repo_root / "data" / "fma_metadata",
        script_dir / "data" / "fma_metadata",
        Path("../data/fma_metadata"),
        Path("data/fma_metadata"),
    ]:
        if path.exists():
            return path

    raise FileNotFoundError(f"FMA metadata directory not found from {data_dir!r}")


def _select_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Return the MFCC-only feature subset used by the classical baseline reference."""
    mfcc_columns = features.columns.get_level_values("feature") == FEATURE_NAME
    selected = features.loc[:, mfcc_columns].copy()
    if selected.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} MFCC features for the baseline, found {selected.shape[1]}."
        )
    selected.columns = ["_".join(col) for col in selected.columns]
    return selected


def _validate_small_dataset(tracks: pd.DataFrame, features: pd.DataFrame, y_all: pd.Series) -> None:
    """Guardrail checks to match the FMA-small baseline setup."""
    if len(tracks) == 0:
        raise ValueError("tracks.csv is empty.")

    small_tracks = tracks[tracks[("set", "subset")] == "small"]
    if len(small_tracks) != EXPECTED_TRACK_COUNT:
        raise ValueError(f"Expected {EXPECTED_TRACK_COUNT} FMA-small tracks, found {len(small_tracks)}.")

    actual_genres = sorted(small_tracks[("track", "genre_top")].dropna().unique().tolist())
    if actual_genres != EXPECTED_GENRES:
        raise ValueError(f"Unexpected genre set: {actual_genres}")

    if small_tracks.index.has_duplicates:
        raise ValueError("Duplicate track IDs found in the FMA-small subset.")

    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} features, found {features.shape[1]}.")

    if len(y_all) != len(features.index):
        raise ValueError("Feature and label counts mismatched after filtering.")

    if not (features.index == y_all.index).all():
        raise ValueError("Feature/label index alignment failed.")

    if y_all.isna().any():
        raise ValueError("Genre labels contain NaN values.")


def load_fma_small_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return FMA-small feature matrix X and matching genre labels y.

    This follows the notebook logic: use only the FMA-small subset and the
    precomputed feature matrix, then align labels to the same track IDs.
    """
    data_path = _resolve_data_dir(data_dir)

    tracks = pd.read_csv(data_path / "tracks.csv", index_col=0, header=[0, 1])
    features = pd.read_csv(data_path / "features.csv", index_col=0, header=[0, 1, 2])
    small_tracks = tracks[tracks[("set", "subset")] == "small"]
    small_track_ids = small_tracks.index

    if small_track_ids.has_duplicates:
        raise ValueError("Duplicate track IDs found in the FMA-small subset.")

    missing_ids = small_track_ids.difference(features.index)
    if len(missing_ids) > 0:
        raise ValueError(f"Missing {len(missing_ids)} FMA-small feature rows in features.csv.")

    X_all = _select_feature_columns(features).loc[small_track_ids]
    y_all = tracks.loc[small_track_ids, ("track", "genre_top")].copy()

    _validate_small_dataset(tracks, X_all, y_all)

    X = X_all.to_numpy(dtype=np.float64)
    y = y_all.to_numpy()

    del features, tracks, small_tracks, X_all, y_all
    return X, y


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split X/y and standardize using training-only statistics.

    This matches the notebook: stratified 80/20 split and scaler fit on train only.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, received shape {X.shape}.")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} != {len(y)}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_baseline_models() -> dict[str, Any]:
    """Return the four notebook-aligned classical ML baseline models."""
    return {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear Support Vector Machine": LinearSVC(random_state=42, max_iter=5000, dual="auto"),
        "Multi-Layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=100,
            early_stopping=True,
            random_state=42,
        ),
    }


def train_baselines(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    """Train each baseline once and store training duration metadata.

    Genre labels are encoded before fitting because the current scikit-learn MLP
    early-stopping implementation can fail when string labels are scored internally.
    """
    models = get_baseline_models()
    encoded_y = y_train
    label_encoder = None

    if np.asarray(y_train).dtype.kind in {"U", "S", "O"}:
        label_encoder = LabelEncoder()
        encoded_y = label_encoder.fit_transform(y_train)

    for model in models.values():
        model.label_encoder_ = label_encoder
        start = time.perf_counter()
        model.fit(X_train, encoded_y)
        model.training_time_ = float(time.perf_counter() - start)
    return models


def evaluate_baselines(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute the notebook metrics for each model and return predictions."""
    baseline_predictions: dict[str, np.ndarray] = {}
    results_records: list[dict[str, Any]] = []

    for name, model in models.items():
        start = time.perf_counter()
        y_pred_encoded = model.predict(X_test)
        pred_time = time.perf_counter() - start

        label_encoder = getattr(model, "label_encoder_", None)
        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(np.asarray(y_pred_encoded))
        else:
            y_pred = np.asarray(y_pred_encoded)

        baseline_predictions[name] = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        macro_precision = precision_score(
            y_test,
            y_pred,
            labels=GENRE_ORDER,
            average="macro",
            zero_division=0,
        )
        macro_recall = recall_score(
            y_test,
            y_pred,
            labels=GENRE_ORDER,
            average="macro",
            zero_division=0,
        )
        macro_f1 = f1_score(
            y_test,
            y_pred,
            labels=GENRE_ORDER,
            average="macro",
            zero_division=0,
        )

        results_records.append(
            {
                "Model": name,
                "Accuracy": float(accuracy),
                "Macro Precision": float(macro_precision),
                "Macro Recall": float(macro_recall),
                "Macro F1": float(macro_f1),
                "Training Time (s)": float(getattr(model, "training_time_", 0.0)),
                "Prediction Time (s)": float(pred_time),
            }
        )

    baseline_results = pd.DataFrame(results_records)
    baseline_results = baseline_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return baseline_results, baseline_predictions


def predict_genre(model: Any, scaler: StandardScaler, feature_vector: Any) -> str:
    """Predict the class label for a single feature vector using the fitted scaler."""
    if scaler is None:
        raise ValueError("A fitted scaler is required for prediction.")

    vector = np.asarray(feature_vector, dtype=np.float64)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    if vector.ndim != 2:
        raise ValueError(f"feature_vector must be 1D or 2D, got shape {vector.shape}.")

    scaled = scaler.transform(vector)
    prediction = model.predict(scaled)
    if len(prediction) != 1:
        raise ValueError(f"Expected a single prediction, got {len(prediction)} values.")

    label_encoder = getattr(model, "label_encoder_", None)
    if label_encoder is not None:
        return str(label_encoder.inverse_transform(np.asarray(prediction))[0])
    return str(prediction[0])


def save_baseline_bundle(models: dict[str, Any], scaler: StandardScaler, output_dir: str | Path = "models/baseline") -> Path:
    """Persist fitted models and the scaler to a small reusable bundle."""
    project_root = Path(__file__).resolve().parents[1]
    target_dir = Path(output_dir).expanduser()
    if not target_dir.is_absolute():
        target_dir = project_root / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = target_dir / "baseline_bundle.joblib"
    payload = {
        "models": models,
        "scaler": scaler,
        "feature_name": FEATURE_NAME,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "genre_order": GENRE_ORDER,
    }
    joblib.dump(payload, bundle_path)
    return bundle_path


def load_baseline_bundle(bundle_path: str | Path = "models/baseline/baseline_bundle.joblib") -> tuple[dict[str, Any], StandardScaler, dict[str, Any]]:
    """Load a saved baseline bundle and return the model dict plus scaler."""
    project_root = Path(__file__).resolve().parents[1]
    path = Path(bundle_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"Baseline bundle not found at {path}.")

    payload = joblib.load(path)
    models = payload.get("models")
    scaler = payload.get("scaler")
    if models is None or scaler is None:
        raise ValueError(f"Bundle at {path} does not contain the expected baseline payload.")
    return models, scaler, payload
