#!/usr/bin/env python3
"""Immunofluorescence colocalization analysis pipeline.

This script computes intensity, topological, and spatial colocalization metrics
for paired `_c1.jpg` and `_c2.jpg` images across three conditions.

Channel convention:
- c1: amyloid deposit
- c2: MAC
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from skimage import color, io, measure
from skimage.filters import threshold_otsu
from skimage.morphology import closing, dilation, disk, erosion, opening
from skimage.transform import resize


# Configuration models
@dataclass(frozen=True)
class SegmentationConfig:
    """Container for segmentation parameters.

    Args:
        threshold_method (str): Thresholding method, one of {"otsu", "quantile"}.
        threshold_quantile (float): Quantile used when threshold_method is "quantile".
        min_object_size (int): Minimum connected-component size in pixels.
        min_hole_size (int): Minimum hole size in pixels to fill.

    Returns:
        SegmentationConfig: Immutable segmentation parameter set.
    """

    threshold_method: str = "otsu"
    threshold_quantile: float = 0.90
    min_object_size: int = 64
    min_hole_size: int = 64


@dataclass(frozen=True)
class SpatialConfig:
    """Container for spatial metric parameters.

    Args:
        max_radius_px (int): Maximum radius in pixels for Ripley K and g(r).
        n_radius_bins (int): Number of radii bins between 1 and max_radius_px.

    Returns:
        SpatialConfig: Immutable spatial metric parameter set.
    """

    max_radius_px: int = 80
    n_radius_bins: int = 20
    max_points_per_channel: int = 2000


@dataclass(frozen=True)
class ProcessingConfig:
    """Container for image-size and runtime control parameters.

    Args:
        max_pixels_per_image (int): Maximum pixel count used for analysis per image.
        random_seed (int): Seed for reproducible point subsampling.

    Returns:
        ProcessingConfig: Immutable processing configuration.
    """

    max_pixels_per_image: int = 4_000_000
    random_seed: int = 42


# Input and output paths
CONDITIONS: dict[str, Path] = {
    "TTR_MAC_Pre_albumine": Path(
        "/workspaces/immunofluorescence-colocalization/data/Amylose fev 2026 Mac TTR Kappa Lambda/TTR MAC Pré albumine"
    ),
    "AL_MAC_Kappa": Path(
        "/workspaces/immunofluorescence-colocalization/data/Amylose fev 2026 Mac TTR Kappa Lambda/AL MAC Kappa"
    ),
    "AL_MAC_Lambda": Path(
        "/workspaces/immunofluorescence-colocalization/data/Amylose fev 2026 Mac TTR Kappa Lambda/AL MAC Lambda"
    ),
}

RESULTS_DIR = Path(
    "/workspaces/immunofluorescence-colocalization/data/Amylose fev 2026 Mac TTR Kappa Lambda/results"
)
FIGURES_DIR = RESULTS_DIR / "figures"
Image.MAX_IMAGE_PIXELS = None


# Core image processing and metric utilities
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale float format.

    Args:
        image (np.ndarray): Input image in grayscale or RGB(A) format.

    Returns:
        np.ndarray: Two-dimensional grayscale image as float64.
    """

    if image.ndim == 2:
        return image.astype(np.float64, copy=False)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        rgb = image[..., :3]
        gray = color.rgb2gray(rgb)
        return gray.astype(np.float64, copy=False)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_to_unit_interval(image: np.ndarray) -> np.ndarray:
    """Normalize image intensities to [0, 1].

    Args:
        image (np.ndarray): Grayscale image array.

    Returns:
        np.ndarray: Normalized image with values in [0, 1].
    """

    image = image.astype(np.float64, copy=False)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))
    span = max_val - min_val
    if span <= 0.0:
        return np.zeros_like(image, dtype=np.float64)
    return (image - min_val) / span


def load_channel(path: Path) -> np.ndarray:
    """Load and normalize one image channel from disk.

    Args:
        path (Path): Path to a channel image file.

    Returns:
        np.ndarray: Normalized grayscale image in [0, 1].
    """

    image = io.imread(path)
    return normalize_to_unit_interval(to_grayscale(image))


def segment_mask(image: np.ndarray, config: SegmentationConfig) -> tuple[np.ndarray, float]:
    """Segment a binary mask from a normalized grayscale image.

    Args:
        image (np.ndarray): Normalized grayscale image in [0, 1].
        config (SegmentationConfig): Segmentation configuration.

    Returns:
        tuple[np.ndarray, float]: Tuple of (binary mask, threshold value).
    """

    if config.threshold_method == "otsu":
        threshold = float(threshold_otsu(image))
    elif config.threshold_method == "quantile":
        threshold = float(np.quantile(image, config.threshold_quantile))
    else:
        raise ValueError(f"Unknown threshold_method: {config.threshold_method}")

    mask = image > threshold
    mask = opening(mask, footprint=disk(1))
    mask = closing(mask, footprint=disk(1))

    if config.min_object_size > 1:
        labels = measure.label(mask)
        areas = np.bincount(labels.ravel())
        keep = areas >= config.min_object_size
        keep[0] = False
        mask = keep[labels]

    if config.min_hole_size > 1:
        inverse = np.logical_not(mask)
        inverse_labels = measure.label(inverse)
        inverse_areas = np.bincount(inverse_labels.ravel())
        border_labels = np.unique(
            np.concatenate(
                [
                    inverse_labels[0, :],
                    inverse_labels[-1, :],
                    inverse_labels[:, 0],
                    inverse_labels[:, -1],
                ]
            )
        )
        fill = inverse_areas < config.min_hole_size
        fill[0] = False
        fill[border_labels] = False
        mask = np.logical_or(mask, fill[inverse_labels])
    return mask.astype(bool), threshold


def resize_pair_if_needed(channel_1: np.ndarray, channel_2: np.ndarray, max_pixels_per_image: int) -> tuple[np.ndarray, np.ndarray]:
    """Downscale a channel pair when image size exceeds a pixel threshold.

    Args:
        channel_1 (np.ndarray): First normalized channel image.
        channel_2 (np.ndarray): Second normalized channel image.
        max_pixels_per_image (int): Maximum allowed pixels per image.

    Returns:
        tuple[np.ndarray, np.ndarray]: Possibly resized channel_1 and channel_2 arrays.
    """

    height, width = channel_1.shape
    total_pixels = height * width
    if total_pixels <= max_pixels_per_image:
        return channel_1, channel_2

    scale = np.sqrt(max_pixels_per_image / float(total_pixels))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))

    resized_1 = resize(channel_1, (new_height, new_width), preserve_range=True, anti_aliasing=True).astype(np.float64)
    resized_2 = resize(channel_2, (new_height, new_width), preserve_range=True, anti_aliasing=True).astype(np.float64)
    return resized_1, resized_2


def pair_files(condition_dir: Path) -> tuple[list[tuple[str, Path, Path]], list[dict[str, Any]]]:
    """Pair c1 and c2 files by shared basename in one condition folder.

    Args:
        condition_dir (Path): Folder containing image files with suffixes `_c1.jpg` and `_c2.jpg`.

    Returns:
        tuple[list[tuple[str, Path, Path]], list[dict[str, Any]]]:
            - Complete pairs as (pair_id, c1_path, c2_path).
            - QC records for complete and incomplete pairing states.
    """

    c1_files = sorted(condition_dir.glob("*_c1.jpg"))
    c2_files = sorted(condition_dir.glob("*_c2.jpg"))

    c1_map = {p.name.replace("_c1.jpg", ""): p for p in c1_files}
    c2_map = {p.name.replace("_c2.jpg", ""): p for p in c2_files}
    all_ids = sorted(set(c1_map) | set(c2_map))

    pairs: list[tuple[str, Path, Path]] = []
    qc_records: list[dict[str, Any]] = []
    for pair_id in all_ids:
        c1_path = c1_map.get(pair_id)
        c2_path = c2_map.get(pair_id)
        if c1_path is not None and c2_path is not None:
            pairs.append((pair_id, c1_path, c2_path))
            qc_records.append(
                {
                    "pair_id": pair_id,
                    "status": "ok",
                    "c1_path": str(c1_path),
                    "c2_path": str(c2_path),
                }
            )
        else:
            qc_records.append(
                {
                    "pair_id": pair_id,
                    "status": "missing_pair",
                    "c1_path": "" if c1_path is None else str(c1_path),
                    "c2_path": "" if c2_path is None else str(c2_path),
                }
            )
    return pairs, qc_records


def pearson_corr(channel_1: np.ndarray, channel_2: np.ndarray) -> float:
    """Compute Pearson correlation between two intensity channels.

    Args:
        channel_1 (np.ndarray): First normalized intensity image.
        channel_2 (np.ndarray): Second normalized intensity image.

    Returns:
        float: Pearson correlation coefficient, or np.nan if undefined.
    """

    x = channel_1.ravel()
    y = channel_2.ravel()
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(pearsonr(x, y).statistic)


def spearman_corr(channel_1: np.ndarray, channel_2: np.ndarray) -> float:
    """Compute Spearman rank correlation between two intensity channels.

    Args:
        channel_1 (np.ndarray): First normalized intensity image.
        channel_2 (np.ndarray): Second normalized intensity image.

    Returns:
        float: Spearman rank correlation coefficient, or np.nan if undefined.
    """

    x = channel_1.ravel()
    y = channel_2.ravel()
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def manders_coefficients(channel_1: np.ndarray, channel_2: np.ndarray, mask_1: np.ndarray, mask_2: np.ndarray) -> tuple[float, float]:
    """Compute Manders overlap coefficients M1 and M2.

    Args:
        channel_1 (np.ndarray): First normalized intensity image.
        channel_2 (np.ndarray): Second normalized intensity image.
        mask_1 (np.ndarray): Binary mask for channel_1 foreground.
        mask_2 (np.ndarray): Binary mask for channel_2 foreground.

    Returns:
        tuple[float, float]: (M1, M2) overlap coefficients.
    """

    sum_1 = float(channel_1.sum())
    sum_2 = float(channel_2.sum())
    m1 = float(channel_1[mask_2].sum() / sum_1) if sum_1 > 0 else float("nan")
    m2 = float(channel_2[mask_1].sum() / sum_2) if sum_2 > 0 else float("nan")
    return m1, m2


def lis_icq(channel_1: np.ndarray, channel_2: np.ndarray) -> float:
    """Compute Li's Intensity Correlation Quotient (ICQ).

    Args:
        channel_1 (np.ndarray): First normalized intensity image.
        channel_2 (np.ndarray): Second normalized intensity image.

    Returns:
        float: Li's ICQ in [-0.5, 0.5].
    """

    x = channel_1.ravel()
    y = channel_2.ravel()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    products = x_centered * y_centered
    return float(np.mean(products > 0.0) - 0.5)


def jaccard_index(mask_1: np.ndarray, mask_2: np.ndarray) -> float:
    """Compute Jaccard index (IoU) between two binary masks.

    Args:
        mask_1 (np.ndarray): First binary mask.
        mask_2 (np.ndarray): Second binary mask.

    Returns:
        float: Jaccard index value in [0, 1], or np.nan if union is empty.
    """

    intersection = np.logical_and(mask_1, mask_2).sum()
    union = np.logical_or(mask_1, mask_2).sum()
    if union == 0:
        return float("nan")
    return float(intersection / union)


def dice_coefficient(mask_1: np.ndarray, mask_2: np.ndarray) -> float:
    """Compute Sorensen-Dice coefficient between two binary masks.

    Args:
        mask_1 (np.ndarray): First binary mask.
        mask_2 (np.ndarray): Second binary mask.

    Returns:
        float: Dice coefficient value in [0, 1], or np.nan when both masks are empty.
    """

    intersection = np.logical_and(mask_1, mask_2).sum()
    denom = mask_1.sum() + mask_2.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * intersection / denom)


def border_engagement_fraction(mask_1: np.ndarray, mask_2: np.ndarray) -> float:
    """Compute fraction of mask_1 border in contact with mask_2.

    Args:
        mask_1 (np.ndarray): Primary binary mask whose border is evaluated.
        mask_2 (np.ndarray): Secondary binary mask checked for border contact.

    Returns:
        float: Fraction of border pixels from mask_1 that touch mask_2.
    """

    border_1 = np.logical_and(mask_1, np.logical_not(erosion(mask_1, footprint=disk(1))))
    touched = np.logical_and(border_1, dilation(mask_2, footprint=disk(1)))
    border_count = int(border_1.sum())
    if border_count == 0:
        return float("nan")
    return float(touched.sum() / border_count)


def centroids_from_mask(mask: np.ndarray) -> np.ndarray:
    """Extract object centroids from a binary mask.

    Args:
        mask (np.ndarray): Binary object mask.

    Returns:
        np.ndarray: Array of centroid coordinates with shape (n_objects, 2).
    """

    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        return np.empty((0, 2), dtype=np.float64)
    return np.array([p.centroid for p in props], dtype=np.float64)


def nearest_neighbor_distances(points_1: np.ndarray, points_2: np.ndarray) -> np.ndarray:
    """Compute nearest-neighbor distances from points_1 to points_2.

    Args:
        points_1 (np.ndarray): Source points with shape (n1, 2).
        points_2 (np.ndarray): Target points with shape (n2, 2).

    Returns:
        np.ndarray: Nearest-neighbor distances for each point in points_1.
    """

    if points_1.size == 0 or points_2.size == 0:
        return np.array([], dtype=np.float64)
    dist_matrix = cdist(points_1, points_2, metric="euclidean")
    return dist_matrix.min(axis=1)


def subsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Subsample point coordinates to keep pairwise distance costs bounded.

    Args:
        points (np.ndarray): Input points with shape (n_points, 2).
        max_points (int): Maximum number of points to keep.
        rng (np.random.Generator): Random generator for reproducible sampling.

    Returns:
        np.ndarray: Original or subsampled points with shape (<=max_points, 2).
    """

    if points.shape[0] <= max_points:
        return points
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def ripley_cross_k(points_1: np.ndarray, points_2: np.ndarray, image_shape: tuple[int, int], radii: np.ndarray) -> np.ndarray:
    """Estimate cross-channel Ripley's K-function K12(r).

    Args:
        points_1 (np.ndarray): Centroids from channel 1 with shape (n1, 2).
        points_2 (np.ndarray): Centroids from channel 2 with shape (n2, 2).
        image_shape (tuple[int, int]): Image shape as (height, width).
        radii (np.ndarray): Radius values in pixels.

    Returns:
        np.ndarray: K12(r) estimates for each radius.
    """

    n1 = points_1.shape[0]
    n2 = points_2.shape[0]
    if n1 == 0 or n2 == 0:
        return np.full(radii.shape, np.nan, dtype=np.float64)

    area = float(image_shape[0] * image_shape[1])
    dist_matrix = cdist(points_1, points_2, metric="euclidean")
    counts = np.array([(dist_matrix <= r).sum() for r in radii], dtype=np.float64)
    return area * counts / float(n1 * n2)


def radial_distribution_function(points_1: np.ndarray, points_2: np.ndarray, image_shape: tuple[int, int], radii: np.ndarray) -> np.ndarray:
    """Estimate cross-pair correlation g(r) from annulus counts.

    Args:
        points_1 (np.ndarray): Centroids from channel 1 with shape (n1, 2).
        points_2 (np.ndarray): Centroids from channel 2 with shape (n2, 2).
        image_shape (tuple[int, int]): Image shape as (height, width).
        radii (np.ndarray): Radius edge values in pixels.

    Returns:
        np.ndarray: g(r) values for annuli between radius edges.
    """

    n1 = points_1.shape[0]
    n2 = points_2.shape[0]
    if n1 == 0 or n2 == 0 or radii.size < 2:
        return np.full((max(radii.size - 1, 1),), np.nan, dtype=np.float64)

    area = float(image_shape[0] * image_shape[1])
    density_2 = n2 / area
    dist_matrix = cdist(points_1, points_2, metric="euclidean")

    g_values: list[float] = []
    for r_lo, r_hi in zip(radii[:-1], radii[1:]):
        annulus_hits = np.logical_and(dist_matrix > r_lo, dist_matrix <= r_hi).sum()
        annulus_area = np.pi * (r_hi**2 - r_lo**2)
        expected_per_point = density_2 * annulus_area
        observed_per_point = annulus_hits / float(n1)
        if expected_per_point <= 0:
            g_values.append(float("nan"))
        else:
            g_values.append(float(observed_per_point / expected_per_point))
    return np.array(g_values, dtype=np.float64)


def safe_mean(values: np.ndarray) -> float:
    """Compute nan-aware mean with empty-array protection.

    Args:
        values (np.ndarray): Numeric array that can include NaN values.

    Returns:
        float: Mean value, or np.nan when no finite values exist.
    """

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def safe_std(values: np.ndarray) -> float:
    """Compute nan-aware standard deviation with empty-array protection.

    Args:
        values (np.ndarray): Numeric array that can include NaN values.

    Returns:
        float: Standard deviation, or np.nan when no finite values exist.
    """

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite, ddof=0))


def plot_diagnostics(
    pair_id: str,
    condition_name: str,
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    mask_1: np.ndarray,
    mask_2: np.ndarray,
    radii: np.ndarray,
    k_values: np.ndarray,
    g_values: np.ndarray,
    output_dir: Path,
) -> None:
    """Save diagnostic plots for one image pair.

    Args:
        pair_id (str): Pair identifier derived from file basename.
        condition_name (str): Condition label.
        channel_1 (np.ndarray): Normalized channel 1 image.
        channel_2 (np.ndarray): Normalized channel 2 image.
        mask_1 (np.ndarray): Segmented channel 1 mask.
        mask_2 (np.ndarray): Segmented channel 2 mask.
        radii (np.ndarray): Radius grid in pixels.
        k_values (np.ndarray): Ripley K values on radii.
        g_values (np.ndarray): g(r) values on annulus bins.
        output_dir (Path): Directory where figure will be saved.

    Returns:
        None: This function writes a PNG file and returns nothing.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = np.zeros((*channel_1.shape, 3), dtype=np.float64)
    overlay[..., 0] = channel_1  # red
    overlay[..., 1] = channel_2  # green

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    ax = axes.ravel()

    ax[0].imshow(channel_1, cmap="Reds")
    ax[0].set_title("Channel c1 (amyloid)")
    ax[1].imshow(channel_2, cmap="Greens")
    ax[1].set_title("Channel c2 (MAC)")
    ax[2].imshow(overlay)
    ax[2].set_title("Intensity overlay")
    ax[3].imshow(mask_1, cmap="gray")
    ax[3].set_title("Mask c1")
    ax[4].imshow(mask_2, cmap="gray")
    ax[4].set_title("Mask c2")

    if np.isfinite(k_values).any():
        ax[5].plot(radii, k_values, label="K12(r)")
    if g_values.size > 0 and np.isfinite(g_values).any():
        bin_centers = (radii[:-1] + radii[1:]) / 2.0
        ax[5].plot(bin_centers, g_values, label="g(r)")
    ax[5].set_title("Spatial metrics")
    ax[5].set_xlabel("Radius (px)")
    ax[5].legend(loc="best")

    for a in ax:
        a.axis("off" if a in ax[:5] else "on")

    fig.suptitle(f"{condition_name} | {pair_id}")
    fig.tight_layout()
    figure_path = output_dir / f"{condition_name}__{pair_id}__diagnostic.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def analyze_pair(
    pair_id: str,
    condition_name: str,
    c1_path: Path,
    c2_path: Path,
    seg_config: SegmentationConfig,
    spatial_config: SpatialConfig,
    processing_config: ProcessingConfig,
    rng: np.random.Generator,
    figures_dir: Path,
) -> dict[str, Any]:
    """Compute all requested colocalization metrics for one image pair.

    Args:
        pair_id (str): Pair identifier derived from filename stem.
        condition_name (str): Condition label.
        c1_path (Path): Path to channel c1 image.
        c2_path (Path): Path to channel c2 image.
        seg_config (SegmentationConfig): Segmentation parameters.
        spatial_config (SpatialConfig): Spatial analysis parameters.
        processing_config (ProcessingConfig): Runtime control parameters.
        rng (np.random.Generator): Random generator for deterministic subsampling.
        figures_dir (Path): Output directory for diagnostic figures.

    Returns:
        dict[str, Any]: Flat dictionary of computed metrics and metadata.
    """

    channel_1 = load_channel(c1_path)
    channel_2 = load_channel(c2_path)

    if channel_1.shape != channel_2.shape:
        raise ValueError(f"Shape mismatch for pair {pair_id}: {channel_1.shape} vs {channel_2.shape}")

    original_shape = channel_1.shape
    channel_1, channel_2 = resize_pair_if_needed(channel_1, channel_2, processing_config.max_pixels_per_image)

    mask_1, threshold_1 = segment_mask(channel_1, seg_config)
    mask_2, threshold_2 = segment_mask(channel_2, seg_config)

    pearson_value = pearson_corr(channel_1, channel_2)
    spearman_value = spearman_corr(channel_1, channel_2)
    m1, m2 = manders_coefficients(channel_1, channel_2, mask_1, mask_2)
    icq = lis_icq(channel_1, channel_2)

    iou = jaccard_index(mask_1, mask_2)
    dice = dice_coefficient(mask_1, mask_2)
    border_fraction = border_engagement_fraction(mask_1, mask_2)

    points_1 = centroids_from_mask(mask_1)
    points_2 = centroids_from_mask(mask_2)
    points_1 = subsample_points(points_1, spatial_config.max_points_per_channel, rng)
    points_2 = subsample_points(points_2, spatial_config.max_points_per_channel, rng)
    nnd = nearest_neighbor_distances(points_1, points_2)

    radii = np.linspace(1.0, float(spatial_config.max_radius_px), spatial_config.n_radius_bins)
    k_values = ripley_cross_k(points_1, points_2, channel_1.shape, radii)
    g_values = radial_distribution_function(points_1, points_2, channel_1.shape, radii)

    plot_diagnostics(
        pair_id=pair_id,
        condition_name=condition_name,
        channel_1=channel_1,
        channel_2=channel_2,
        mask_1=mask_1,
        mask_2=mask_2,
        radii=radii,
        k_values=k_values,
        g_values=g_values,
        output_dir=figures_dir,
    )

    return {
        "condition": condition_name,
        "pair_id": pair_id,
        "c1_path": str(c1_path),
        "c2_path": str(c2_path),
        "height_px": channel_1.shape[0],
        "width_px": channel_1.shape[1],
        "orig_height_px": original_shape[0],
        "orig_width_px": original_shape[1],
        "threshold_c1": threshold_1,
        "threshold_c2": threshold_2,
        "pearson_r": pearson_value,
        "spearman_rho": spearman_value,
        "manders_m1": m1,
        "manders_m2": m2,
        "li_icq": icq,
        "jaccard_iou": iou,
        "dice_f1": dice,
        "border_engagement_fraction": border_fraction,
        "n_objects_c1": int(points_1.shape[0]),
        "n_objects_c2": int(points_2.shape[0]),
        "nnd_mean_px": safe_mean(nnd),
        "nnd_std_px": safe_std(nnd),
        "nnd_median_px": float(np.nanmedian(nnd)) if nnd.size else float("nan"),
        "ripley_k_mean": safe_mean(k_values),
        "ripley_k_auc": float(np.trapezoid(np.nan_to_num(k_values, nan=0.0), radii)),
        "ripley_k_rmax": float(k_values[-1]) if k_values.size else float("nan"),
        "gr_mean": safe_mean(g_values),
        "gr_peak": float(np.nanmax(g_values)) if np.isfinite(g_values).any() else float("nan"),
    }


def summarize_by_condition(per_image_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-image metrics into condition-level aggregates.

    Args:
        per_image_df (pd.DataFrame): Per-image metric table.

    Returns:
        pd.DataFrame: Condition-level summary with mean, std, and sample count.
    """

    numeric_cols = per_image_df.select_dtypes(include=[np.number]).columns.tolist()
    grouped = per_image_df.groupby("condition", dropna=False)[numeric_cols]

    mean_df = grouped.mean(numeric_only=True).add_suffix("_mean")
    std_df = grouped.std(numeric_only=True).add_suffix("_std")
    n_df = grouped.size().to_frame(name="n_pairs")

    summary = pd.concat([n_df, mean_df, std_df], axis=1).reset_index()
    return summary


def analyze_condition(
    condition_name: str,
    condition_dir: Path,
    seg_config: SegmentationConfig,
    spatial_config: SpatialConfig,
    processing_config: ProcessingConfig,
    rng: np.random.Generator,
    figures_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run pairwise colocalization analysis for one condition directory.

    Args:
        condition_name (str): Human-readable condition name.
        condition_dir (Path): Directory containing `_c1.jpg` and `_c2.jpg` files.
        seg_config (SegmentationConfig): Segmentation configuration.
        spatial_config (SpatialConfig): Spatial metric configuration.
        processing_config (ProcessingConfig): Runtime control parameters.
        rng (np.random.Generator): Random generator for deterministic subsampling.
        figures_dir (Path): Base figure output directory.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Per-image metrics for complete pairs.
            - Pairing QC table with complete and missing pair statuses.
    """

    pairs, qc_records = pair_files(condition_dir)
    qc_df = pd.DataFrame(qc_records)
    if not qc_df.empty:
        qc_df.insert(0, "condition", condition_name)
    else:
        qc_df = pd.DataFrame(columns=["condition", "pair_id", "status", "c1_path", "c2_path"])

    records: list[dict[str, Any]] = []
    condition_fig_dir = figures_dir / condition_name
    for pair_id, c1_path, c2_path in pairs:
        record = analyze_pair(
            pair_id=pair_id,
            condition_name=condition_name,
            c1_path=c1_path,
            c2_path=c2_path,
            seg_config=seg_config,
            spatial_config=spatial_config,
            processing_config=processing_config,
            rng=rng,
            figures_dir=condition_fig_dir,
        )
        records.append(record)

    per_image_df = pd.DataFrame(records)
    return per_image_df, qc_df


# End-to-end execution helpers
def run_all_conditions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute full colocalization workflow for all configured conditions.

    Args:
        None: This function uses global condition and output configurations.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - Per-image metrics table.
            - Condition-level summary table.
            - Pairing QC table.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    seg_config = SegmentationConfig()
    spatial_config = SpatialConfig()
    processing_config = ProcessingConfig()
    rng = np.random.default_rng(processing_config.random_seed)

    all_per_image: list[pd.DataFrame] = []
    all_qc: list[pd.DataFrame] = []

    for condition_name, condition_dir in CONDITIONS.items():
        if not condition_dir.exists():
            all_qc.append(
                pd.DataFrame(
                    [
                        {
                            "condition": condition_name,
                            "pair_id": "",
                            "status": "missing_condition_dir",
                            "c1_path": "",
                            "c2_path": "",
                        }
                    ]
                )
            )
            continue
        per_image_df, qc_df = analyze_condition(
            condition_name=condition_name,
            condition_dir=condition_dir,
            seg_config=seg_config,
            spatial_config=spatial_config,
            processing_config=processing_config,
            rng=rng,
            figures_dir=FIGURES_DIR,
        )
        if not per_image_df.empty:
            all_per_image.append(per_image_df)
        all_qc.append(qc_df)

    per_image_all = pd.concat(all_per_image, ignore_index=True) if all_per_image else pd.DataFrame()
    qc_all = pd.concat(all_qc, ignore_index=True) if all_qc else pd.DataFrame()
    summary = summarize_by_condition(per_image_all) if not per_image_all.empty else pd.DataFrame()

    per_image_path = RESULTS_DIR / "colocalization_per_image.csv"
    summary_path = RESULTS_DIR / "colocalization_per_condition_summary.csv"
    qc_path = RESULTS_DIR / "colocalization_qc.csv"

    per_image_all.to_csv(per_image_path, index=False)
    summary.to_csv(summary_path, index=False)
    qc_all.to_csv(qc_path, index=False)

    return per_image_all, summary, qc_all


if __name__ == "__main__":
    per_image_df, summary_df, qc_df = run_all_conditions()

    print("=== Colocalization workflow complete ===")
    print(f"Per-image results: {len(per_image_df)} rows")
    print(f"Condition summary: {len(summary_df)} rows")
    print(f"QC records: {len(qc_df)} rows")

    if not qc_df.empty:
        status_counts = qc_df["status"].value_counts(dropna=False)
        print("\nQC status counts:")
        print(status_counts.to_string())
