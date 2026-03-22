# Cat vs Dog Faces Lab Sequence

This repo includes three connected Jupyter notebooks in [labs](labs):

- [lab1_numpy_cats_vs_dogs.ipynb](labs/lab1_numpy_cats_vs_dogs.ipynb)
- [lab2_pandas_cats_vs_dogs.ipynb](labs/lab2_pandas_cats_vs_dogs.ipynb)
- [lab3_pytorch_cats_vs_dogs.ipynb](labs/lab3_pytorch_cats_vs_dogs.ipynb)

Shared visualization helpers live in [lab_utils/visualization.py](lab_utils/visualization.py). They cover:

- image galleries for NumPy exploration
- class-balance and error-rate plots for Pandas analysis
- training curves and reference-style feature-map grids for PyTorch

The lessons use a lightweight curated Kaggle dataset of cat and dog faces that students can download directly.

- [Student dataset - Kaggle](https://www.kaggle.com/datasets/duongtranhai/cats-dogs-faces-small/data)

The notebooks expect a small prepared teaching subset at:

```text
data/cats_dogs_faces_small/
  train/
    cat/
    dog/
  val/
    cat/
    dog/
  test/
    cat/
    dog/
```

## Download the Dataset

Use [scripts/download_animal_faces.py](scripts/download_animal_faces.py) to download the already-cleaned dataset directly from Kaggle.

Recommended steps:

1. Install project dependencies, for example with `uv sync`.
2. Make sure Kaggle credentials are available through `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. Run:

```bash
uv run python scripts/download_animal_faces.py --force
```

That script will:

- download `duongtranhai/cats-dogs-faces-small` from Kaggle
- extract the cleaned dataset into `data/cats_dogs_faces_small/`
- verify that `train`, `val`, `test`, and `metadata.csv` are present
- verify that the extracted dataset contains images

## Recommended setup

- Use the included curated `64x64` dataset for fast CPU-friendly labs

## Notebook flow

- Lab 1 uses NumPy to inspect images as arrays and build a nearest-centroid baseline
- Lab 2 uses Pandas for metadata analysis, split thinking, and NumPy error analysis
- Lab 3 uses PyTorch to train a small binary classifier and compare it against the NumPy baseline
