# Cats vs Dogs Lab Sequence

This repo includes three connected Jupyter notebooks in [labs](labs):

- [lab1_numpy_cats_vs_dogs.ipynb](labs/lab1_numpy_cats_vs_dogs.ipynb)
- [lab2_pandas_cats_vs_dogs.ipynb](labs/lab2_pandas_cats_vs_dogs.ipynb)
- [lab3_pytorch_cats_vs_dogs.ipynb](labs/lab3_pytorch_cats_vs_dogs.ipynb)

Shared visualization helpers live in [lab_utils/visualization.py](lab_utils/visualization.py). They cover:

- image galleries for NumPy exploration
- class-balance and error-rate plots for Pandas analysis
- training curves and feature-map grids for PyTorch

## Dataset layout by lab

- Lab 1 uses the original Kaggle Dogs vs Cats competition images for basic NumPy image operations.
- Labs 2 and 3 use the lightweight curated cat-and-dog-faces subset for fast Pandas and PyTorch work.

## Lab 1: Original Cats vs Dogs

Use the original competition dataset extracted into:

```text
data/dogs_vs_cats_original/
  train/
    cat.0.jpg
    dog.0.jpg
    ...
```

Download it with [scripts/download_original_dogs_vs_cats.py](scripts/download_original_dogs_vs_cats.py):

```bash
uv run python scripts/download_original_dogs_vs_cats.py --force
```

That script will:

- download the original Kaggle `dogs-vs-cats` competition files
- extract the dataset into `data/dogs_vs_cats_original/`
- verify that the `train/` folder contains both cat and dog images

Kaggle note: you must accept the competition rules on the competition page before the API download will work.

Lab 1 saves its predictions to:

```text
artifacts/lab1_numpy_original_predictions_<student_id>.csv
```

This Lab 1 notebook is now a standalone NumPy lab. Labs 2 and 3 do not depend on this original-dataset prediction file.

## Labs 2 and 3: Curated Faces Dataset

The later labs use a lightweight curated Kaggle dataset of cat and dog faces that students can download directly.

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

- Lab 1 uses NumPy to inspect original cats-and-dogs images as arrays and build a tiny hand-crafted baseline
- Lab 2 uses Pandas for curated metadata analysis, split thinking, and error analysis
- Lab 3 uses PyTorch to train a simple CNN on the curated face dataset and visualize stage-1 and stage-2 feature maps

## Student-specific variants

- Set the same `STUDENT_ID` in the first code cell of all three notebooks
- That value is used as the random seed for sample selection, split shuffling, batch order, and the PyTorch feature-map example
- Lab 1 saves `artifacts/lab1_numpy_original_predictions_<student_id>.csv`
- Labs 2 and 3 continue to use the curated face dataset flow
