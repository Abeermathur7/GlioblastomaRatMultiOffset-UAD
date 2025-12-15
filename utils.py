from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib.colors import Normalize, ListedColormap, to_rgb
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# =============================================================================
# Basic metrics
# =============================================================================


def reconstruction_error(recon: np.ndarray, target: np.ndarray):
    """
    Per-sample mean squared error (MSE) across feature dimension.
    """
    return np.mean((recon - target) ** 2, axis=1)


def roc_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    pos_label: int = 1,
) -> Dict[str, float]:
    """
    Run a ROC analysis and pick the threshold that maximizes Youden's J.

    Returns a dict with:
      - fpr, tpr, thresholds
      - auc
      - best_threshold, sensitivity, specificity
      - youden_j
      - fraction_flagged (percent above threshold)
      - threshold_percentile (where the threshold sits in the score CDF)
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Youden's J = TPR - FPR (equivalently: sensitivity + specificity - 1)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    best_threshold = thresholds[best_idx]
    sensitivity = tpr[best_idx]
    specificity = 1 - fpr[best_idx]
    youden_j = sensitivity + specificity - 1

    fraction_flagged = (scores >= best_threshold).mean() * 100.0
    threshold_percentile = np.mean(scores <= best_threshold) * 100.0

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": roc_auc,
        "best_threshold": best_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_j": youden_j,
        "fraction_flagged": fraction_flagged,
        "threshold_percentile": threshold_percentile,
    }


def summary_text(metrics: Dict[str, float]) -> str:
    """
    Compact text summary of key ROC numbers (nice for logs / captions).
    """
    return (
        f"AUC: {metrics['auc']:.3f}\n"
        f"Optimal threshold (Youden's J): {metrics['best_threshold']:.4f}\n"
        f"Sensitivity: {metrics['sensitivity']:.3f}\n"
        f"Specificity: {metrics['specificity']:.3f}\n"
        f"Fraction flagged as anomaly: {metrics['fraction_flagged']:.1f}%\n"
        f"Threshold percentile: {metrics['threshold_percentile']:.1f}th\n"
    )


# =============================================================================
# ROC / PR plots
# =============================================================================


def plot_roc_clean(
    metrics: Dict[str, float],
    *,
    figsize: Tuple[float, float] = (6, 4),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    ROC curve with the Youden-optimal point highlighted and annotated.
    """
    fpr = metrics["fpr"]
    tpr = metrics["tpr"]
    roc_auc = metrics["auc"]

    best_threshold = metrics["best_threshold"]
    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]
    youden_j = metrics["youden_j"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=1.5)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="0.6", label="Chance")

    # Best operating point on ROC (FPR = 1-specificity, TPR = sensitivity)
    best_fpr = 1 - specificity
    best_tpr = sensitivity
    ax.scatter(
        best_fpr,
        best_tpr,
        color="red",
        zorder=5,
        label=f"Optimal thresh = {best_threshold:.4f}",
    )
    ax.annotate(
        f"Youden's J = {youden_j:.3f}",
        xy=(best_fpr, best_tpr),
        xytext=(best_fpr + 0.1, best_tpr - 0.08),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        fontsize=7,
    )

    ax.set_xlabel("False Positive Rate", labelpad=2)
    ax.set_ylabel("True Positive Rate", labelpad=2)
    ax.set_title("CAE Anomaly Detection ROC", pad=6)

    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.grid(alpha=0.2, linewidth=0.5)

    # Clean frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(width=0.6, length=3)

    plt.tight_layout()

    if save_path:
        dpi = 300 if save_path.lower().endswith((".pdf", ".svg")) else 600
        fig.savefig(save_path, dpi=dpi)

    return fig, ax


def plot_roc_pr(
    y_true,
    scores,
    *,
    pos_label: int = 1,
    figsize: tuple[float, float] = (3.5, 2.0),
    save_path: str | None = None,
    show: bool = True,
):
    """
    Side-by-side ROC and precision–recall curves for an anomaly score.
    """
    fpr, tpr, _ = roc_curve(y_true, scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, scores, pos_label=pos_label)
    ap_score = average_precision_score(y_true, scores)

    plt.rcParams.update(
        {
            "figure.figsize": figsize,
            "figure.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "lines.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    fig, axes = plt.subplots(1, 2, sharey=False)

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color="#E69F00", label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("(a) ROC curve", loc="left", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # PR
    ax = axes[1]
    ax.plot(rec, prec, color="#56B4E9", label=f"AP = {ap_score:.3f}")
    ax.set_title("(b) Precision–Recall", loc="left", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        dpi = 300 if save_path.lower().endswith((".pdf", ".svg")) else 600
        # fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


# =============================================================================
# Cropping helpers
# =============================================================================


def _union_crop(imgs, pad: int = 3):
    """
    Compute a single crop (row slice, col slice) that covers the non-zero support
    of *all* 2D images passed in.
    """
    row_ranges, col_ranges = [], []

    for img in imgs:
        mask = img > 0
        if not mask.any():
            # Fall back to full image if there's no support.
            row_ranges.append((0, img.shape[0]))
            col_ranges.append((0, img.shape[1]))
            continue

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0 = max(rows[0] - pad, 0)
        r1 = min(rows[-1] + pad + 1, img.shape[0])
        c0 = max(cols[0] - pad, 0)
        c1 = min(cols[-1] + pad + 1, img.shape[1])
        row_ranges.append((r0, r1))
        col_ranges.append((c0, c1))

    r0 = min(r[0] for r in row_ranges)
    r1 = max(r[1] for r in row_ranges)
    c0 = min(c[0] for c in col_ranges)
    c1 = max(c[1] for c in col_ranges)
    return slice(r0, r1), slice(c0, c1)


# =============================================================================
# Anomaly / binary maps (vertical stacks)
# =============================================================================


def plot_anomaly_maps(
    anom_volume: np.ndarray,
    offset_idx: int = 0,
    rat_indices=(0, 1, 2),
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "inferno",
    figsize=(4, 10),
    pad_crop: int = 3,
):
    """
    Three anomaly maps stacked vertically (one per rat) with a shared colorbar.
    """
    imgs = []
    for ri in rat_indices:
        if anom_volume.ndim == 4:
            imgs.append(anom_volume[ri, ..., offset_idx])
        elif anom_volume.ndim == 3:
            imgs.append(anom_volume[ri])
        else:
            raise ValueError(f"Unexpected ndim {anom_volume.ndim}")

    rs, cs = _union_crop(imgs, pad=pad_crop)

    stacked = np.stack(imgs, axis=0)
    if vmin is None:
        positive = stacked[stacked > 0]
        vmin = float(positive.min()) if positive.size else 0.0
    if vmax is None:
        vmax = float(stacked.max())

    labels = ["a", "b", "c"]
    fig = plt.figure(figsize=figsize, facecolor="black")
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.12], hspace=0.05)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    for i, img in enumerate(imgs):
        ax = axes[i]
        ax.set_facecolor("black")
        cropped = img[rs, cs]
        im = ax.imshow(
            cropped,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        ax.axis("off")
        ax.text(
            0.02,
            0.95,
            labels[i],
            transform=ax.transAxes,
            color="white",
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

    cax = fig.add_subplot(gs[3, 0])
    ticks = np.linspace(vmin, vmax, 3)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.set_label("Anomaly Score", color="white", fontsize=15)
    cbar.ax.tick_params(color="white", labelcolor="white", labelsize=15)
    cbar.outline.set_edgecolor("white")

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_binary_maps(
    binary_volume: np.ndarray,
    offset_idx: int = 0,
    rat_indices=(0, 1, 2),
    active_color: str = "#FF6B6B",
    figsize=(4, 10),
    pad_crop: int = 3,
):
    """
    Same layout as plot_anomaly_maps, but for binary masks.
    """
    masks = []
    for ri in rat_indices:
        if binary_volume.ndim == 4:
            m = binary_volume[ri, ..., offset_idx].astype(bool)
        elif binary_volume.ndim == 3:
            m = binary_volume[ri].astype(bool)
        else:
            raise ValueError(f"Unexpected ndim {binary_volume.ndim}")
        masks.append(m.astype(int))

    rs, cs = _union_crop(masks, pad=pad_crop)

    labels = ["a", "b", "c"]
    cmap_bin = ListedColormap(["black", active_color])

    fig = plt.figure(figsize=figsize, facecolor="black")
    gs = fig.add_gridspec(3, 1, hspace=0.05)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    for i, mask in enumerate(masks):
        ax = axes[i]
        ax.set_facecolor("black")
        cropped = mask[rs, cs]
        ax.imshow(
            cropped,
            cmap=cmap_bin,
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="nearest",
        )
        ax.axis("off")
        ax.text(
            0.02,
            0.95,
            labels[i],
            transform=ax.transAxes,
            color="white",
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle(
        "Binary Detection Map", y=0.98, color="white", fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
    return fig, axes


# =============================================================================
# Metabolite maps (3×5 grids) and combined layouts
# =============================================================================


def plot_offset_grid(SDF1, SDF4, CDF1):
    """
    Show SDF1 / SDF4 / CDF1 in a 3×5 grid (rows = dataset, cols = metabolite).

    Each column has its own color scale based on the 1st–99th percentiles.
    """
    order = [0, 1, 2, 3, 4]
    titles = ["Water", "MT", "APT", "Amine", "NOE"]
    row_labels = ["SDF1", "SDF4", "CDF1"]

    fig_width, fig_height = 12, 8
    left_margin, right_margin = 0.05, 0.95
    bar_h = 0.02
    bar_bottom = 0.04

    # Flip L/R once for all inputs (kept exactly as original).
    vols = [SDF1[:, ::-1, :], SDF4[:, ::-1, :], CDF1[:, ::-1, :]]

    # Tight crop where any volume has non-zero-ish signal.
    mask = np.zeros(vols[0].shape[:2], bool)
    for v in vols:
        mask |= np.any(v > np.finfo(v.dtype).eps, axis=2)

    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    vols_c = [v[r0:r1, c0:c1, :] for v in vols]

    # Percentile limits per channel across all three volumes.
    vmins, vmaxs = {}, {}
    for ch in order:
        arr = np.concatenate([vol[:, :, ch].ravel() for vol in vols_c])
        vmins[ch], vmaxs[ch] = np.percentile(arr, [1, 99])

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="black")
    gs = fig.add_gridspec(3, 5, wspace=0.02, hspace=0.02)

    for i, vol in enumerate(vols_c):
        for j, ch in enumerate(order):
            ax = fig.add_subplot(gs[i, j], facecolor="black")
            ax.imshow(
                vol[:, :, ch],
                cmap="inferno",
                norm=Normalize(vmins[ch], vmaxs[ch]),
                aspect="equal",
            )
            ax.axis("off")

            if i == 0:
                ax.set_title(titles[j], color="white", fontsize=12, weight="bold", pad=6)
            if j == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_labels[i],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    color="white",
                    fontsize=12,
                    weight="bold",
                )

    # Five small colorbars underneath (one per column).
    ncol = len(order)
    total = right_margin - left_margin
    col_w = total / ncol
    shrink = 0.8
    bar_w = col_w * shrink
    offset = (col_w - bar_w) / 2

    for idx, ch in enumerate(order):
        col_left = left_margin + idx * col_w
        left = col_left + offset
        cax = fig.add_axes([left, bar_bottom, bar_w, bar_h], facecolor="black")

        sm = ScalarMappable(cmap="inferno", norm=Normalize(vmins[ch], vmaxs[ch]))
        sm.set_array([])

        v0, v2 = vmins[ch], vmaxs[ch]
        v1 = 0.5 * (v0 + v2)
        ticks = [v0, v1, v2]

        cb = plt.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
        cb.ax.set_xticklabels([f"{t:.2f}" for t in ticks])

        cb.outline.set_edgecolor("white")
        cax.xaxis.set_tick_params(
            color="white",
            labelsize=8,
            length=3,
            direction="out",
        )
        for lbl in cax.get_xticklabels():
            lbl.set_color("white")

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.97,
        bottom=bar_bottom + bar_h + 0.01,
        wspace=0.02,
        hspace=0.02,
    )

    return None


def plot_combined_grid_and_anomaly(
    SDF1,
    SDF4,
    CDF1,
    anom_volume,
    offset_idx: int = 0,
    rat_indices=(0, 1, 2),
    pad_crop: int = 3,
    cmap: str = "inferno",
    fig_size=(16, 8),
):
    """
    Put SDF1/SDF4/CDF1 metabolite maps and anomaly maps into one figure.

    Left:  3×5 metabolite grid (rows = SDF1/SDF4/CDF1, cols = metabolites).
    Right: anomaly maps for selected rats, with a shared colorbar.
    """

    def _local_union_crop(images, pad: int = 3):
        # Same logic/thresholding as original (eps-based).
        mask = np.zeros_like(images[0], dtype=bool)
        for img in images:
            mask |= img > np.finfo(img.dtype).eps

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        r0, r1 = rows[0] - pad, rows[-1] + 1 + pad
        c0, c1 = cols[0] - pad, cols[-1] + 1 + pad

        r0 = max(r0, 0)
        c0 = max(c0, 0)
        return slice(r0, r1), slice(c0, c1)

    channels = [0, 1, 2, 3, 4]
    titles = ["Water", "MT", "APT", "Amine", "NOE"]
    row_labels = ["SDF1", "SDF4", "CDF1"]

    vols = [SDF1[:, ::-1, :], SDF4[:, ::-1, :], CDF1[:, ::-1, :]]

    # Tight crop for metabolite maps.
    mask = np.zeros(vols[0].shape[:2], dtype=bool)
    for v in vols:
        mask |= np.any(v > np.finfo(v.dtype).eps, axis=2)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    vols_c = [v[r0:r1, c0:c1, :] for v in vols]

    # Percentile scaling per channel.
    vmins, vmaxs = {}, {}
    for ch in channels:
        arr = np.concatenate([vol[:, :, ch].ravel() for vol in vols_c])
        vmins[ch], vmaxs[ch] = np.percentile(arr, [1, 99])

    # Pull anomaly slices (3D or 4D supported).
    anomaly_imgs = []
    for ri in rat_indices:
        if anom_volume.ndim == 4:
            anomaly_imgs.append(anom_volume[ri, :, :, offset_idx])
        elif anom_volume.ndim == 3:
            anomaly_imgs.append(anom_volume[ri])
        else:
            raise ValueError("anom_volume must be 3D or 4D")

    rs, cs = _local_union_crop(anomaly_imgs, pad=pad_crop)
    anomaly_cropped = [img[rs, cs] for img in anomaly_imgs]

    # Fixed anomaly scaling (kept exactly as original).
    vmin_a = 0.0
    vmax_a = 0.15

    fig = plt.figure(figsize=fig_size, facecolor="black")

    top, bottom = 0.96, 0.20
    hspace = 0.02

    gs_left = fig.add_gridspec(
        3,
        5,
        left=0.03,
        right=0.72,
        top=top,
        bottom=bottom,
        wspace=0.02,
        hspace=hspace,
    )
    gs_right = fig.add_gridspec(
        3,
        1,
        left=0.76,
        right=0.98,
        top=top,
        bottom=bottom,
        hspace=hspace,
    )

    # --- metabolite maps (left) ---
    for i, vol in enumerate(vols_c):
        for j, ch in enumerate(channels):
            ax = fig.add_subplot(gs_left[i, j], facecolor="black")
            ax.imshow(
                vol[:, :, ch],
                cmap=cmap,
                norm=Normalize(vmins[ch], vmaxs[ch]),
                aspect="equal",
            )
            ax.axis("off")

            if i == 0:
                ax.set_title(titles[j], color="white", fontsize=12, weight="bold", pad=6)
            if j == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_labels[i],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    color="white",
                    fontsize=12,
                    weight="bold",
                )

    # --- colorbars under metabolite maps (5 separate) ---
    bar_bottom = 0.04
    bar_h = 0.02
    ncol = len(channels)
    total_width = 0.68 - 0.05
    col_width = total_width / ncol
    shrink = 0.8
    bar_width = col_width * shrink
    offset = (col_width - bar_width) / 2

    for idx, ch in enumerate(channels):
        col_left = 0.05 + idx * col_width
        left_ax = col_left + offset
        cax = fig.add_axes([left_ax, bar_bottom, bar_width, bar_h], facecolor="black")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmins[ch], vmaxs[ch]))
        sm.set_array([])

        v0, v2 = vmins[ch], vmaxs[ch]
        v1 = 0.5 * (v0 + v2)
        ticks = [v0, v1, v2]

        cb = plt.colorbar(sm, cax=cax, orientation="horizontal", ticks=ticks)
        cb.ax.set_xticklabels([f"{t:.2f}" for t in ticks])
        cb.outline.set_edgecolor("white")
        cb.ax.tick_params(
            color="white",
            labelcolor="white",
            labelsize=8,
            length=3,
            direction="out",
        )

    # --- anomaly maps (right) ---
    right_axes = [fig.add_subplot(gs_right[i, 0]) for i in range(3)]
    for ax, img in zip(right_axes, anomaly_cropped):
        ax.imshow(
            img,
            cmap=cmap,
            vmin=vmin_a,
            vmax=vmax_a,
            aspect="equal",
            interpolation="nearest",
        )
        ax.axis("off")

    # Shared anomaly colorbar (bottom-right).
    right_left, right_right = 0.76, 0.98
    right_center = 0.5 * (right_left + right_right)

    cax_right = fig.add_axes(
        [right_center - bar_width / 2, bar_bottom, bar_width, bar_h],
        facecolor="black",
    )

    ticks = np.array([vmin_a, 0.5 * (vmin_a + vmax_a), vmax_a])

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin_a, vmax_a), cmap=cmap),
        cax=cax_right,
        orientation="horizontal",
        ticks=ticks,
    )
    cbar.set_label("Anomaly score", color="white", fontsize=12)
    cbar.ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(
        color="white",
        labelcolor="white",
        labelsize=10,
        length=3,
        direction="out",
    )
    cbar.outline.set_edgecolor("white")

    return fig


# =============================================================================
# Tumor detection overlays
# =============================================================================


def plot_tumor_detection(
    bin_img_cae: np.ndarray,
    tumor_voxel_img: np.ndarray,
    save_path: str = "tumor_detection.png",
):
    """
    Compare automated CAE detections to manual tumor masks for three rats.

    For each row (rat), the figure shows:
      - CAE mask
      - ground truth mask
      - overlap map with Dice and sensitivity.
    """
    auto_mask = bin_img_cae[..., 0].astype(bool)
    truth_mask = tumor_voxel_img[..., 0].astype(bool)

    rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "figure.dpi": 300,
        }
    )

    colors = {"auto": "#FF6B6B", "truth": "#4ECDC4", "overlap": "#FFE66D"}
    titles = [
        "Automated Detection\n(CAE)",
        "Manual Segmentation\n(Ground Truth)",
        "Overlap Analysis",
    ]
    panels = ["a", "b", "c"]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(6.8, 6),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        facecolor="none",
    )

    for r in range(3):
        a = auto_mask[r]
        t = truth_mask[r]
        ov = np.logical_and(a, t)

        dice = (2 * ov.sum()) / (a.sum() + t.sum()) if (a.sum() + t.sum()) else 0
        sens = ov.sum() / t.sum() if t.sum() else 0

        rgba_auto = (*to_rgb(colors["auto"]), 0.8)
        rgba_truth = (*to_rgb(colors["truth"]), 0.8)
        rgba_overlap = (*to_rgb(colors["overlap"]), 0.8)

        for c in range(3):
            ax = axes[r, c]
            ax.set_facecolor("none")

            H, W = a.shape
            canvas = np.zeros((H, W, 4), dtype=float)

            if c == 0:
                canvas[a] = rgba_auto
            elif c == 1:
                canvas[t] = rgba_truth
            else:
                canvas[a & ~t] = rgba_auto
                canvas[t & ~a] = rgba_truth
                canvas[ov] = rgba_overlap
                ax.text(
                    0.02,
                    0.98,
                    f"Dice: {dice:.3f}\nOverlap: {sens:.1%}",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor="gray",
                        linewidth=0.5,
                    ),
                )

            ax.imshow(canvas, interpolation="none")
            ax.axis("off")

            if r == 0:
                ax.set_title(titles[c], fontsize=8, pad=5)
                ax.text(
                    -0.15,
                    1.05,
                    panels[c],
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                )

    handles = [
        Patch(color=colors["auto"], label="Automated only"),
        Patch(color=colors["truth"], label="Manual only"),
        Patch(color=colors["overlap"], label="Overlap"),
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        fontsize=7,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.text(
        0.5,
        0.98,
        "Post-GAD 52 Offset Sampled Tumor Detection ",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    # plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

    return fig, axes


# =============================================================================
# Z-spectra reconstruction summary
# =============================================================================


def plot_zspec_recon_and_residuals_scatter(
    orig,
    rec,
    y_true,
    *,
    offset_axis: np.ndarray | None = None,
    healthy_label: int = 0,
    tumor_label: int = 1,
    figsize: tuple[float, float] = (3.5, 3.5),
    save_path: str | None = None,
    show: bool = True,
):
    """
\   """

    def _to_numpy_2d(x):
        import numpy as _np

        try:
            import torch as _torch

            if isinstance(x, _torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass

        x = _np.array(x)
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        return x

    orig = _to_numpy_2d(orig)
    rec = _to_numpy_2d(rec)
    y = np.array(y_true).ravel()

    assert orig.shape == rec.shape, f"Shape mismatch: {orig.shape} vs {rec.shape}"
    assert orig.ndim == 2

    n_offsets = orig.shape[1]
    # if offset_axis is None:
    #     offset_axis = np.linspace(-4, 4, n_offsets)

    healthy_inds = np.where(y == healthy_label)[0]
    tumor_inds = np.where(y == tumor_label)[0]

    def _stats(idxs):
        o = orig[idxs]
        r = rec[idxs]
        return {
            "orig_mean": o.mean(axis=0),
            "rec_mean": r.mean(axis=0),
            "mae": np.abs(o - r).mean(axis=0),
            "mse": ((o - r) ** 2).mean(axis=0),
        }

    h = _stats(healthy_inds)
    t = _stats(tumor_inds)

    # Styling (kept exactly)
    plt.rcParams.update(
        {
            "figure.figsize": figsize,
            "figure.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
        }
    )

    col_spec = {"orig": "#6A51A3", "rec": "#009E73"}
    col_err = {"mae": "#E69F00", "mse": "#0072B2"}

    fig, axes = plt.subplots(2, 2, sharex=True, sharey="row")
    s = 6  # marker size

    # (a) Healthy spectra
    ax = axes[0, 0]
    sc1 = ax.scatter(offset_axis, h["orig_mean"], color=col_spec["orig"], s=s)
    sc2 = ax.scatter(offset_axis, h["rec_mean"], color=col_spec["rec"], s=s)
    ax.set_title("(a) Healthy Z-spectrum", loc="left")
    ax.set_ylabel("1-S$_{sat}$/S$_0$")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (b) Tumor spectra
    ax = axes[0, 1]
    ax.scatter(offset_axis, t["orig_mean"], color=col_spec["orig"], s=s)
    ax.scatter(offset_axis, t["rec_mean"], color=col_spec["rec"], s=s)
    ax.set_title("(b) Tumor Z-spectrum", loc="left")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (c) Healthy residuals
    ax = axes[1, 0]
    sc3 = ax.scatter(offset_axis, h["mae"], color=col_err["mae"], s=s)
    sc4 = ax.scatter(offset_axis, h["mse"], color=col_err["mse"], s=s)
    ax.set_title("(c) Healthy residuals", loc="left")
    ax.set_xlabel("Offset (ppm)")
    ax.set_ylabel("Error")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # (d) Tumor residuals
    ax = axes[1, 1]
    ax.scatter(offset_axis, t["mae"], color=col_err["mae"], s=s)
    ax.scatter(offset_axis, t["mse"], color=col_err["mse"], s=s)
    ax.set_title("(d) Tumor residuals", loc="left")
    ax.set_xlabel("Offset (ppm)")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend (kept placement/entries)
    fig.legend(
        [sc1, sc2, sc3, sc4],
        ["Original", "Reconstruction", "MAE", "MSE"],
        loc="upper left",
        ncol=1,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )

    fig.suptitle(
        f"CAE reconstruction and residual analysis of rat Z-spectra\n({len(offset_axis)} Offset)",
        fontsize=9,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)

    if save_path:
        dpi = 300 if save_path.lower().endswith((".pdf", ".svg")) else 600
        # fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


# =============================================================================
# T1 + anomaly + masks overlay grid (2 rats)
# =============================================================================


def plot_masks_over_t1_slices(
    t1_slices,
    auto_masks,
    truth_masks,
    *,
    anom_maps=None,
    anom_offset_idx=0,
    anom_cmap="inferno",
    anom_alpha=0.65,
    anom_vmin=0,
    anom_vmax=0.07,
    offset_number=52,
    titles=(
        "T1",
        "Anomaly Score",
        "CAE Model Output",
        "Manual Segmentation",
        "Overlap Analysis",
    ),
    colors=None,
    save_path="tumor_detection.png",
):
    """
    Multi-panel summary of T1 slices, anomaly scores, and masks.

    Rows: 2 rats
    Cols:
      1) T1
      2) anomaly heatmap (optional) + ONE colorbar (top row only)
      3) automated mask
      4) manual mask
      5) overlap
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import to_rgb, Normalize
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if colors is None:
        colors = {
            "auto": "#FF6B6B",
            "truth": "#4ECDC4",
            "overlap": "#FFE66D",
        }

    # Only plot first two rats (kept exactly).
    t1_slices = list(t1_slices)
    n_rats = 2
    assert len(t1_slices) >= n_rats, "Need at least 2 T1 slices (one per rat)."

    auto_masks = auto_masks.astype(bool)[:n_rats]
    truth_masks = truth_masks.astype(bool)[:n_rats]

    H, W = auto_masks[0].shape
    for i in range(n_rats):
        assert t1_slices[i].shape == (H, W), f"Rat {i}: T1 shape mismatch."
        assert auto_masks[i].shape == (H, W)
        assert truth_masks[i].shape == (H, W)

    show_anomaly = anom_maps is not None

    # ---- anomaly prep ----
    if show_anomaly:
        anomaly_slices = []
        for i in range(n_rats):
            anom = anom_maps[i]
            if anom.ndim == 3:
                anom = anom[..., anom_offset_idx]
            assert anom.shape == (H, W), f"Rat {i}: anomaly map shape mismatch."
            anomaly_slices.append(anom)
        anomaly_slices = np.stack(anomaly_slices, axis=0)

        if anom_vmin is None:
            pos = anomaly_slices[anomaly_slices > 0]
            anom_vmin = float(pos.min()) if pos.size else 0.0
        if anom_vmax is None:
            anom_vmax = float(anomaly_slices.max())

        norm = Normalize(vmin=anom_vmin, vmax=anom_vmax)
        cmap = cm.get_cmap(anom_cmap)

    # ---- RGBA colors for overlays ----
    color_auto = (*to_rgb(colors["auto"]), 0.80)
    color_truth = (*to_rgb(colors["truth"]), 0.80)
    color_overlap = (*to_rgb(colors["overlap"]), 0.80)

    num_cols = 5 if show_anomaly else 4
    width_ratios = [1] * num_cols

    fig, axes = plt.subplots(
        nrows=n_rats,
        ncols=num_cols,
        figsize=(10.2, 4.4) if show_anomaly else (8.6, 4.4),
        gridspec_kw={"wspace": 0.03, "hspace": 0.03, "width_ratios": width_ratios},
        facecolor="black",
    )

    if n_rats == 1:
        axes = axes[None, :]

    # Robust grayscale window based on rat 0 only (kept exactly).
    v = t1_slices[0][np.isfinite(t1_slices[0])]
    gray_min, gray_max = np.percentile(v, [2, 98])

    def _show_t1(ax, img):
        ax.imshow(img, cmap="gray", vmin=gray_min, vmax=gray_max, interpolation="none")
        ax.set_facecolor("black")
        ax.axis("off")

    for r in range(n_rats):
        t1 = t1_slices[r]
        am = auto_masks[r]
        tm = truth_masks[r]

        both = am & tm
        auto_only = am & ~tm
        truth_only = tm & ~am

        denom = am.sum() + tm.sum()
        dice = (2 * both.sum() / denom) if denom else 0.0

        c_t1 = 0
        c_anom = 1 if show_anomaly else None
        c_auto = 2 if show_anomaly else 1
        c_truth = 3 if show_anomaly else 2
        c_ovlp = 4 if show_anomaly else 3

        # --- T1 ---
        _show_t1(axes[r, c_t1], t1)

        # --- anomaly heatmap overlay (with ONE colorbar on top row only) ---
        if show_anomaly:
            ax = axes[r, c_anom]
            _show_t1(ax, t1)

            heat = cmap(norm(anomaly_slices[r]))
            heat[..., 3] = anom_alpha
            ax.imshow(heat, interpolation="nearest")

            if r == 0:
                cax = inset_axes(
                    ax,
                    width="70%",
                    height="6%",
                    loc="upper center",
                    borderpad=0.01,
                )
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
                cb.ax.tick_params(labelsize=6, colors="white", length=2)
                cb.outline.set_edgecolor("white")
                for spine in cax.spines.values():
                    spine.set_edgecolor("white")

        # --- automated mask ---
        ax = axes[r, c_auto]
        _show_t1(ax, t1)
        overlay = np.zeros((H, W, 4), dtype=float)
        overlay[am] = color_auto
        ax.imshow(overlay, interpolation="none")

        # --- manual mask ---
        ax = axes[r, c_truth]
        _show_t1(ax, t1)
        overlay[:] = 0
        overlay[tm] = color_truth
        ax.imshow(overlay, interpolation="none")

        # --- overlap panel ---
        ax = axes[r, c_ovlp]
        _show_t1(ax, t1)
        overlay[:] = 0
        overlay[auto_only] = color_auto
        overlay[truth_only] = color_truth
        overlay[both] = color_overlap
        ax.imshow(overlay, interpolation="none")

        ax.text(
            0.02,
            0.98,
            f"Dice: {dice:.3f}",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="black",
                alpha=0.85,
                edgecolor="white",
                linewidth=0.6,
            ),
        )

    # Column titles (top row only).
    for c in range(num_cols):
        axes[0, c].set_title(titles[c], fontsize=11, color="white", pad=8)

    # Legend
    legend_handles = [
        Patch(color=colors["auto"], label="Automated only"),
        Patch(color=colors["truth"], label="Manual only"),
        Patch(color=colors["overlap"], label="Overlap"),
    ]
    leg = fig.legend(
        handles=legend_handles,
        ncol=3,
        fontsize=10,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
    )
    for txt in leg.get_texts():
        txt.set_color("white")

    # Big title
    fig.text(
        0.5,
        0.98,
        f"Post-GAD {offset_number} Offset Sampled Tumor Detection",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    fig.subplots_adjust(top=0.82, bottom=0.12)

    # fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    return fig, axes
