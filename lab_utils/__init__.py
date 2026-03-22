"""Shared helpers for the cats-vs-dogs lab sequence."""

from .visualization import (
    arrange_images_on_grid,
    extract_feature_maps,
    plot_centroid_heatmap,
    plot_class_balance,
    plot_error_rate_by_group,
    plot_feature_maps_like_reference,
    plot_feature_vector,
    plot_numeric_distribution,
    plot_prediction_gallery,
    plot_training_history,
    show_image_gallery,
    show_tensor_batch,
)

__all__ = [
    "arrange_images_on_grid",
    "extract_feature_maps",
    "plot_centroid_heatmap",
    "plot_class_balance",
    "plot_error_rate_by_group",
    "plot_feature_maps_like_reference",
    "plot_feature_vector",
    "plot_numeric_distribution",
    "plot_prediction_gallery",
    "plot_training_history",
    "show_image_gallery",
    "show_tensor_batch",
]
