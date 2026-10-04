# Real Time Music Genre Classification

A machine learning mini-project for automatic music genre classification using the **FMA-small** dataset. The project compares classical machine-learning baselines with a SongNet-style deep-learning model and provides a local **Gradio** interface for testing audio files.

## Team Members

| Name | SRN |
|---|---|
| Vishruta | PES1UG24CS538 |
| Trisha | PES1UG24CS503 |

## Problem Statement

The objective of this project is to classify a 30-second audio clip into one of eight music genres:

- Electronic
- Experimental
- Folk
- Hip-Hop
- Instrumental
- International
- Pop
- Rock

The project uses two complementary approaches:

1. **Classical ML baselines** trained using precomputed FMA audio features.
2. **SongNet-style deep learning** trained on mel-spectrograms generated from raw FMA-small audio.

A Gradio application is used to upload an audio file and view predictions from the trained models.

## Dataset

The project uses **FMA-small (Free Music Archive)**.

FMA-small contains:

- 8,000 audio tracks
- Approximately 30 seconds per track
- 8 balanced genres
- 1,000 tracks per genre

Two FMA downloads are used:

- `fma_small` — raw audio files used by the SongNet pipeline
- `fma_metadata` — track labels and precomputed features used by the project

The dataset is not included in this repository.

## Repository Structure

```text
Music_Genre_Classification/
│
├── models/
│   ├── songnet_paper.pt
│   └── baseline/
│       └── baseline_bundle.joblib
│
├── notebooks/
│   ├── baseline.ipynb
│   ├── baseline.py
│   ├── baseline_accuracy_comparison.png
│   ├── baseline_best_model_confusion_matrix.png
│   ├── baseline_confusion_matrices.png
│   ├── baseline_macro_f1_comparison.png
│   ├── class_distribution.png
│   └── songnet_results.ipynb
│
├── src/
│   ├── app.py
│   ├── baseline.py
│   ├── predict.py
│   └── songnet.py
│
├── FMA_test_case_manifest.txt
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Prerequisites

A Python installation with `pip` and `venv` support is required.

The project dependencies are listed in `requirements.txt` and include PyTorch, Gradio, librosa, NumPy, pandas, scikit-learn, SciPy and joblib.

## Setup

Clone the repository and move into the project directory:

```powershell
git clone <repository-url>
cd Music_Genre_Classification
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The application uses `src` as a Python package. If `src/__init__.py` is not already present, create it once:

```powershell
New-Item src\__init__.py -ItemType File
```

## Dataset Setup

Create a `data` directory in the project root:

```text
Music_Genre_Classification/
└── data/
    ├── fma_small/
    └── fma_metadata/
```

After downloading and extracting the FMA files, the structure should look similar to:

```text
data/
├── fma_small/
│   ├── 000/
│   ├── 001/
│   ├── 002/
│   └── ...
│
└── fma_metadata/
    ├── tracks.csv
    ├── features.csv
    ├── genres.csv
    └── ...
```

Do not place an extra nested `fma_small` or `fma_metadata` directory inside these folders.

## Running the Application

The trained SongNet and baseline model files are already included in the repository under `models/`.

From the project root, activate the virtual environment and run:

```powershell
python -m src.app
```

Gradio will start a local web server and display a local URL in the terminal. Open that URL in a browser.

In the application:

1. Drag and drop or upload an audio file.
2. Start the analysis.
3. View the SongNet genre prediction and probability distribution.
4. View the predictions produced by the classical ML models.

No model training is performed when the application starts.

## Notebooks

The project contains two main notebooks.

### SongNet

`notebooks/songnet_results.ipynb`

Contains the SongNet preprocessing, model training/evaluation workflow, training curves, confusion matrix and prediction results.

### Classical ML Baselines

`notebooks/baseline.ipynb`

Contains the classical machine-learning pipeline and comparison of K-Nearest Neighbors, Logistic Regression, Linear Support Vector Machine and Multi-Layer Perceptron.

Open the notebooks using Jupyter from the project environment.

## Implementation Overview

### SongNet Pipeline

```text
Audio
  ↓
Mel-Spectrogram
  ↓
Temporal 1D CNN Blocks
  ↓
Time-wise Genre Logits
  ↓
Temporal Averaging
  ↓
8-Genre Prediction
```

The reusable model definition is in `src/songnet.py`, inference is handled by `src/predict.py`, and the trained checkpoint is `models/songnet_paper.pt`.

### Classical ML Pipeline

The baseline models use precomputed FMA MFCC features and are implemented in `src/baseline.py`. The saved model bundle is `models/baseline/baseline_bundle.joblib`.

### Gradio Demo

The local application is implemented in `src/app.py` and connects the SongNet inference pipeline and classical baseline models into one interface.

## References

1. Chi Zhang, Yue Zhang and Chen Chen, **“SongNet: Real-time Music Classification,”** Stanford CS229, 2018.  
   https://cs229.stanford.edu/proj2018/report/53.pdf

2. Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst and Xavier Bresson, **“FMA: A Dataset For Music Analysis,”** ISMIR, 2017.  
   https://arxiv.org/abs/1612.01840

3. FMA Dataset Repository:  
   https://github.com/mdeff/fma
