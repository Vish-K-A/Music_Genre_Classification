from pathlib import Path
import pandas as pd

# Resolve repository and data paths dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["src", "notebooks"] else SCRIPT_DIR
DATA_DIR = REPO_ROOT / "data" / "fma_metadata"

if not DATA_DIR.exists():
    # Fallback if working directory differs
    for candidate in [Path("../data/fma_metadata"), Path("data/fma_metadata"), Path("fma_metadata")]:
        if candidate.exists():
            DATA_DIR = candidate
            break

tracks = pd.read_csv(
    DATA_DIR / "tracks.csv",
    index_col=0,
    header=[0, 1]
)

genres = pd.read_csv(
    DATA_DIR / "genres.csv"
)

print("Tracks shape:", tracks.shape)
print("Genres shape:", genres.shape)

print("\nTrack subsets:")
print(tracks[("set", "subset")].value_counts())

print("\nGenre columns:")
print(genres.columns.tolist())

# Subset for FMA-small
small_tracks = tracks[tracks[("set", "subset")] == "small"]
print("\nFMA-small tracks:", len(small_tracks))
