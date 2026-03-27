# Cats vs Dogs Lab Sequence

This repo includes three connected Jupyter notebooks in [labs](labs):

- [lab1_numpy_cats_vs_dogs.ipynb](labs/lab1_numpy_cats_vs_dogs.ipynb)
- [lab2_pandas_cats_vs_dogs.ipynb](labs/lab2_pandas_cats_vs_dogs.ipynb)
- [lab3_pytorch_cats_vs_dogs.ipynb](labs/lab3_pytorch_cats_vs_dogs.ipynb)

Shared visualization helpers live in [lab_utils/visualization.py](lab_utils/visualization.py). They cover:

- image galleries for NumPy exploration
- class-balance and error-rate plots for Pandas analysis
- training curves and feature-map grids for PyTorch

## Dataset

All three labs use the lightweight curated Kaggle dataset of cat and dog faces.

Dataset available at: .data/cats_dogs_faces_small/

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

## Notebook flow

- Lab 1 uses NumPy to inspect cat and dog face images as arrays and build a small hand-crafted feature matrix
- Lab 2 uses Pandas for curated metadata analysis, split thinking, and error analysis
- Lab 3 uses PyTorch to train a simple CNN on the same face dataset and visualize stage-1 and stage-2 feature maps

## Student-specific variants

- Set the same `STUDENT_ID` in the first code cell of all three notebooks
- That value is used as the random seed for sample selection, split shuffling, batch order, and the PyTorch feature-map example
- Lab 1 uses `STUDENT_ID` to choose a reproducible subset and preview order
