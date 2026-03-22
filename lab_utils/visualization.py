from __future__ import annotations

import math
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _as_numpy_image(image: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    return array


def show_image_gallery(
    images: Sequence[np.ndarray | Sequence[float]],
    titles: Sequence[str] | None = None,
    *,
    ncols: int = 4,
    figsize: tuple[float, float] = (12, 6),
    suptitle: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Display a small gallery of RGB or grayscale images."""
    if not images:
        raise ValueError("Expected at least one image.")

    n_images = len(images)
    ncols = max(1, min(ncols, n_images))
    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= n_images:
            continue

        image = _as_numpy_image(images[idx])
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            ax.imshow(np.squeeze(image), cmap="gray")
        else:
            ax.imshow(image)

        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx])

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def show_tensor_batch(
    images: np.ndarray,
    labels: Sequence[int] | None = None,
    *,
    class_names: Sequence[str] | None = None,
    max_items: int = 8,
    ncols: int = 4,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, np.ndarray]:
    """Display a batch of channel-first tensors."""
    batch = np.asarray(images)
    max_items = min(max_items, batch.shape[0])
    gallery = [batch[idx] for idx in range(max_items)]

    titles = None
    if labels is not None:
        label_array = np.asarray(labels)
        titles = []
        for idx in range(max_items):
            label_value = int(label_array[idx])
            if class_names is not None:
                titles.append(class_names[label_value])
            else:
                titles.append(str(label_value))

    return show_image_gallery(gallery, titles=titles, ncols=ncols, figsize=figsize)


def plot_feature_vector(
    features: Sequence[float],
    feature_names: Sequence[str] | None = None,
    *,
    title: str = "Feature Vector",
    figsize: tuple[float, float] = (12, 3.5),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 1D feature vector as a bar chart."""
    values = np.asarray(features, dtype=float)
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(len(values))]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(values)), values, color="#4C6FFF")
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_centroid_heatmap(
    centroids: Sequence[Sequence[float]],
    feature_names: Sequence[str],
    *,
    class_names: Sequence[str] = ("cat", "dog"),
    title: str = "Class Centroids",
    figsize: tuple[float, float] = (12, 2.8),
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize class centroids as a compact heatmap."""
    matrix = np.asarray(centroids, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(class_names)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig, ax


def plot_prediction_gallery(
    image_paths: Sequence,
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    load_image_fn,
    *,
    max_items: int = 8,
    ncols: int = 4,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, np.ndarray]:
    """Show a labeled gallery of predictions."""
    max_items = min(max_items, len(image_paths))
    images = [load_image_fn(path) for path in image_paths[:max_items]]
    titles = [
        f"true={true_labels[idx]}\npred={pred_labels[idx]}"
        for idx in range(max_items)
    ]
    return show_image_gallery(images, titles=titles, ncols=ncols, figsize=figsize)


def plot_class_balance(
    frame,
    *,
    split_col: str = "split",
    label_col: str = "label",
    title: str = "Class Balance by Split",
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a grouped bar chart of label counts by split."""
    summary = frame.groupby([split_col, label_col]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=figsize)
    summary.plot(kind="bar", ax=ax, color=["#4C6FFF", "#FF7A59"])
    ax.set_title(title)
    ax.set_ylabel("Images")
    ax.set_xlabel(split_col)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_numeric_distribution(
    frame,
    *,
    column: str,
    group_col: str = "label",
    bins: int = 20,
    title: str | None = None,
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay simple histograms for a numeric column."""
    fig, ax = plt.subplots(figsize=figsize)
    for group_name, group_frame in frame.groupby(group_col):
        ax.hist(group_frame[column], bins=bins, alpha=0.45, label=str(group_name))
    ax.set_title(title or f"{column} by {group_col}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_error_rate_by_group(
    frame,
    *,
    group_col: str,
    correct_col: str = "correct_numpy",
    title: str | None = None,
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot error rate per group as a bar chart."""
    summary = 1.0 - frame.groupby(group_col)[correct_col].mean().sort_index()
    fig, ax = plt.subplots(figsize=figsize)
    summary.plot(kind="bar", ax=ax, color="#FF7A59")
    ax.set_title(title or f"Error Rate by {group_col}")
    ax.set_ylabel("Error rate")
    ax.set_xlabel(group_col)
    ax.set_ylim(0.0, min(1.0, max(0.05, float(summary.max()) + 0.05)))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_training_history(
    history,
    *,
    epoch_col: str = "epoch",
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot training and validation loss/accuracy curves."""
    if hasattr(history, "to_dict"):
        records = history.to_dict("records")
    else:
        records = list(history)

    epochs = [row[epoch_col] for row in records]
    train_loss = [row["train_loss"] for row in records]
    val_loss = [row["val_loss"] for row in records]
    train_acc = [row["train_acc"] for row in records]
    val_acc = [row["val_acc"] for row in records]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train")
    axes[1].plot(epochs, val_acc, marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    return fig, axes


def arrange_images_on_grid(
    images: Sequence[np.ndarray],
    grid_size: tuple[int, int],
    *,
    gap: int = 0,
    background_value: int = 0,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    border_width: int = 0,
    border_color: str | tuple[float, float, float] = "#FFFFFF",
) -> np.ndarray:
    """
    Arrange grayscale images on a colored grid.

    This follows the same overall layout idea as the tiled feature-map display in
    Stephen Welch's AlexNet notebook: each channel is shown as its own tile and
    mapped through a colormap before being assembled into one large image.
    """
    if not images:
        raise ValueError("Expected at least one image to arrange.")

    if isinstance(border_color, str):
        hex_color = border_color.lstrip("#")
        border_rgb = tuple(int(hex_color[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
    else:
        border_rgb = border_color

    cmap = plt.get_cmap(cmap_name)
    rows, cols = grid_size
    image_height, image_width = np.asarray(images[0]).shape

    tile_height = image_height + 2 * border_width
    tile_width = image_width + 2 * border_width

    canvas_height = rows * tile_height + (rows - 1) * gap
    canvas_width = cols * tile_width + (cols - 1) * gap
    canvas = np.full((canvas_height, canvas_width, 3), background_value / 255.0, dtype=np.float32)

    max_tiles = rows * cols
    for idx, image in enumerate(images[:max_tiles]):
        tile = np.asarray(image, dtype=np.float32)
        lower = tile.min() if vmin is None else vmin
        upper = tile.max() if vmax is None else vmax

        clipped = np.clip(tile, lower, upper)
        if upper > lower:
            normalized = (clipped - lower) / (upper - lower)
        else:
            normalized = np.zeros_like(clipped)

        colored = cmap(normalized)[..., :3]
        row = idx // cols
        col = idx % cols
        top = row * (tile_height + gap)
        left = col * (tile_width + gap)

        if border_width > 0:
            canvas[top:top + tile_height, left:left + tile_width, :] = border_rgb
            top += border_width
            left += border_width

        canvas[top:top + image_height, left:left + image_width, :] = colored

    return canvas


def extract_feature_maps(feature_extractor, image_tensor, *, layer_up_to: int | None = None, device=None):
    """Run an image through a convolutional feature extractor and return channel maps."""
    import torch

    module = feature_extractor
    if layer_up_to is not None:
        try:
            module = feature_extractor[:layer_up_to]
        except TypeError as exc:
            raise TypeError("layer_up_to requires a sliceable module such as nn.Sequential.") from exc

    batch = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
    if batch.ndim != 4:
        raise ValueError("Expected image_tensor with shape (C, H, W) or (B, C, H, W).")

    target_device = device
    if target_device is None:
        try:
            first_param = next(module.parameters())
            target_device = first_param.device
        except StopIteration:
            target_device = torch.device("cpu")

    with torch.no_grad():
        activations = module(batch.to(target_device))

    if activations.ndim != 4:
        raise ValueError("Expected convolutional activations with shape (B, C, H, W).")

    return activations.detach().cpu()[0]


def plot_feature_maps_like_reference(
    feature_maps,
    *,
    grid_size: tuple[int, int] | None = None,
    gap: int = 2,
    background_value: int = 255,
    cmap_name: str = "viridis",
    vmin: float = -0.6,
    vmax: float = 0.45,
    border_width: int = 0,
    border_color: str | tuple[float, float, float] = "#948979",
    figsize: tuple[float, float] = (12, 12),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Visualize feature maps using the same tiled-grid look as the AlexNet notebook.

    The maps are scaled by their positive maximum before being laid out on a grid,
    matching the overall display pattern used in the referenced implementation.
    """
    maps = np.asarray(feature_maps, dtype=np.float32)
    if maps.ndim == 4:
        maps = maps[0]
    if maps.ndim != 3:
        raise ValueError("Expected feature_maps with shape (C, H, W) or (B, C, H, W).")

    positive_max = float(maps.max())
    scaled_maps = maps / positive_max if positive_max > 0 else maps.copy()

    if grid_size is None:
        cols = max(1, math.ceil(math.sqrt(scaled_maps.shape[0])))
        rows = math.ceil(scaled_maps.shape[0] / cols)
        grid_size = (rows, cols)

    rows, cols = grid_size
    tile_count = min(rows * cols, scaled_maps.shape[0])
    grid_image = arrange_images_on_grid(
        [scaled_maps[idx] for idx in range(tile_count)],
        grid_size=grid_size,
        gap=gap,
        background_value=background_value,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        border_width=border_width,
        border_color=border_color,
    )

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.imshow(grid_image)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax, grid_image
