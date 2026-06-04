"""
visualise.py – Reusable plotting functions for streaming ML workflows.

All functions work standalone, inside pipeline logs, or in demo
notebooks. Every function accepts a save_path argument to write
the figure to disk, and returns the matplotlib Figure object for
inline display in Jupyter.

Required plots (per spec)
-------------------------
plot_metric_over_time(metric_values, title, ylabel)
    Plot a metric (e.g. accuracy) across streaming chunks.

compare_models(metric1, metric2, labels)
    Compare two models on the same streaming metric.

plot_predictions_vs_ground_truth(y_true, y_pred)
    Visualise predictions vs actuals on the latest chunk.

Additional plots
----------------
plot_confusion_matrix(matrix, classes, ...)
plot_learning_curve(n_samples_list, train_scores, val_scores, ...)
plot_histogram(counts, edges, ...)
plot_decision_boundary(model, X, y, ...)

Usage
-----
    from numcompute_stream import visualise

    fig = visualise.plot_metric_over_time(
        accuracy_history,
        title="Accuracy over Chunks",
        ylabel="Accuracy",
        save_path="outputs/accuracy.png",
    )
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ── colour palette ─────────────────────────────────────────────────────────
_COLORS = [
    "#2563EB",   # blue
    "#F97316",   # orange
    "#16A34A",   # green
    "#DC2626",   # red
    "#7C3AED",   # purple
    "#0891B2",   # cyan
]
_GRID_ALPHA  = 0.25
_LINE_WIDTH  = 2.0
_MARKER_SIZE = 5


# ─────────────────────────────────────────────────────────────────────────────
# 1. plot_metric_over_time  (spec required)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_over_time(
    metric_values,
    title="Metric over Time",
    ylabel="Value",
    xlabel="Chunk",
    color=None,
    smoothing=None,
    save_path=None,
    figsize=(9, 4),
):
    """
    Plot a single metric across streaming chunks.

    Parameters
    ----------
    metric_values : array-like, shape (n_chunks,)
        Metric value recorded after each chunk.
    title : str, default='Metric over Time'
    ylabel : str, default='Value'
    xlabel : str, default='Chunk'
    color : str or None
        Line colour. Defaults to the first palette colour.
    smoothing : int or None
        Moving-average window size. None = no smoothing.
    save_path : str or None
        If given, saves the figure to this path.
    figsize : tuple, default=(9, 4)

    Returns
    -------
    matplotlib.figure.Figure
    """
    values = np.asarray(metric_values, dtype=float)
    x      = np.arange(1, len(values) + 1)
    c      = color or _COLORS[0]

    fig, ax = plt.subplots(figsize=figsize)

    # raw line (semi-transparent)
    ax.plot(
        x, values,
        color=c, alpha=0.35, linewidth=1.0,
        label=ylabel,
    )
    # markers
    ax.scatter(x, values, color=c, s=_MARKER_SIZE ** 2, zorder=3)

    # optional moving-average smoothing
    if smoothing and smoothing > 1 and len(values) >= smoothing:
        kernel   = np.ones(smoothing) / smoothing
        smoothed = np.convolve(values, kernel, mode="valid")
        xs       = x[smoothing - 1:]
        ax.plot(
            xs, smoothed,
            color=c, linewidth=_LINE_WIDTH,
            label=f"{ylabel} (MA-{smoothing})",
        )
        ax.legend(fontsize=9)

    _style_ax(ax, title, xlabel, ylabel)
    _set_metric_ylim(ax, ylabel, values)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. compare_models  (spec required)
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(
    metric1,
    metric2,
    labels,
    title="Model Comparison",
    ylabel="Metric",
    xlabel="Chunk",
    save_path=None,
    figsize=(9, 4),
):
    """
    Compare two models on the same streaming metric.

    Parameters
    ----------
    metric1 : array-like, shape (n_chunks,)
        Metric history for the first model.
    metric2 : array-like, shape (n_chunks,)
        Metric history for the second model.
    labels : list[str], length 2
        Display names for the two models, e.g. ['DecisionTree', 'RF'].
    title : str, default='Model Comparison'
    ylabel : str, default='Metric'
    xlabel : str, default='Chunk'
    save_path : str or None
    figsize : tuple, default=(9, 4)

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(labels) < 2:
        raise ValueError(
            f"labels must have at least 2 entries, got {len(labels)}"
        )

    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)
    x1 = np.arange(1, len(m1) + 1)
    x2 = np.arange(1, len(m2) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        x1, m1,
        color=_COLORS[0], linewidth=_LINE_WIDTH,
        marker="o", markersize=_MARKER_SIZE,
        label=labels[0],
    )
    ax.plot(
        x2, m2,
        color=_COLORS[1], linewidth=_LINE_WIDTH,
        marker="s", markersize=_MARKER_SIZE,
        label=labels[1],
    )

    # shade the area between the two curves
    min_len = min(len(m1), len(m2))
    ax.fill_between(
        np.arange(1, min_len + 1),
        m1[:min_len], m2[:min_len],
        alpha=0.08, color="gray",
    )

    _style_ax(ax, title, xlabel, ylabel)
    _set_metric_ylim(ax, ylabel, np.concatenate([m1, m2]))
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. plot_predictions_vs_ground_truth  (spec required)
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title="Predictions vs Ground Truth",
    sample_limit=200,
    save_path=None,
    figsize=(11, 4),
):
    """
    Visualise predictions against ground-truth labels on the latest chunk.

    Produces a two-panel figure:
        Left  – side-by-side bar chart of class counts (true vs predicted).
        Right – dot plot showing correct (✓) and incorrect (✗) predictions
                for the first ``sample_limit`` samples.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_pred : array-like, shape (n_samples,)
    title : str, default='Predictions vs Ground Truth'
    sample_limit : int, default=200
        Maximum samples shown in the dot plot.
    save_path : str or None
    figsize : tuple, default=(11, 4)

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred length mismatch: "
            f"{len(y_true)} vs {len(y_pred)}"
        )

    classes    = np.union1d(y_true, y_pred)
    acc        = float(np.mean(y_true == y_pred))
    n_show     = min(len(y_true), sample_limit)

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(1, 2, width_ratios=[1, 2], figure=fig)
    ax_bar = fig.add_subplot(gs[0])
    ax_dot = fig.add_subplot(gs[1])

    # ── left panel: class count bar chart ────────────────────────────────
    x_pos  = np.arange(len(classes))
    width  = 0.35
    true_counts = np.array([np.sum(y_true == c) for c in classes])
    pred_counts = np.array([np.sum(y_pred == c) for c in classes])

    ax_bar.bar(
        x_pos - width / 2, true_counts,
        width=width, color=_COLORS[0], label="True", alpha=0.85,
    )
    ax_bar.bar(
        x_pos + width / 2, pred_counts,
        width=width, color=_COLORS[1], label="Predicted", alpha=0.85,
    )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([str(c) for c in classes])
    ax_bar.set_xlabel("Class")
    ax_bar.set_ylabel("Count")
    ax_bar.set_title("Class Distribution")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", alpha=_GRID_ALPHA)

    # ── right panel: per-sample correct / incorrect dot plot ──────────────
    yt_show = y_true[:n_show]
    yp_show = y_pred[:n_show]
    correct = yt_show == yp_show
    x_idx   = np.arange(n_show)

    ax_dot.scatter(
        x_idx[correct], yt_show[correct],
        marker="o", s=18, color=_COLORS[2],
        alpha=0.7, label="Correct", zorder=2,
    )
    ax_dot.scatter(
        x_idx[~correct], yt_show[~correct],
        marker="x", s=28, color=_COLORS[3],
        alpha=0.9, label="Incorrect", zorder=3,
    )
    ax_dot.set_xlabel("Sample index")
    ax_dot.set_ylabel("True label")
    ax_dot.set_title(
        f"Per-sample predictions  (acc={acc:.3f}, n={n_show})"
    )
    ax_dot.legend(fontsize=8, loc="upper right")
    ax_dot.grid(alpha=_GRID_ALPHA)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. plot_confusion_matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    matrix,
    classes,
    title="Confusion Matrix",
    normalise=False,
    cmap="Blues",
    save_path=None,
    figsize=(6, 5),
):
    """
    Heatmap of a confusion matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_classes, n_classes)
        Rows = true labels, columns = predicted labels.
    classes : array-like
        Class label names.
    title : str
    normalise : bool
        If True, show row-normalised proportions instead of counts.
    cmap : str, default='Blues'
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    matrix = np.asarray(matrix, dtype=float)
    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix   = matrix / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=figsize)
    im      = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
    ax.set_yticklabels([str(c) for c in classes])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = f"{matrix[i,j]:.2f}" if normalise else f"{int(matrix[i,j])}"
            ax.text(
                j, i, val,
                ha="center", va="center", fontsize=9,
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. plot_learning_curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curve(
    n_samples_list,
    train_scores,
    val_scores=None,
    metric_name="Accuracy",
    title="Learning Curve",
    save_path=None,
    figsize=(9, 4),
):
    """
    Plot train (and optionally validation) metric vs samples seen.

    Parameters
    ----------
    n_samples_list : array-like
        Cumulative sample counts at each evaluation point.
    train_scores : array-like
    val_scores : array-like or None
    metric_name : str
    title : str
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        n_samples_list, train_scores,
        color=_COLORS[0], linewidth=_LINE_WIDTH,
        marker="o", markersize=_MARKER_SIZE, label="Train",
    )
    if val_scores is not None:
        ax.plot(
            n_samples_list, val_scores,
            color=_COLORS[1], linewidth=_LINE_WIDTH,
            marker="s", markersize=_MARKER_SIZE, label="Validation",
        )

    _style_ax(ax, title, "Samples Seen", metric_name)
    _set_metric_ylim(ax, metric_name, np.asarray(train_scores))
    ax.legend(fontsize=9)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. plot_histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_histogram(
    counts,
    edges,
    title="Streaming Histogram",
    xlabel="Value",
    color=None,
    save_path=None,
    figsize=(8, 4),
):
    """
    Bar plot of a streaming histogram (from StreamingHistogram.compute()).

    Parameters
    ----------
    counts : array-like
    edges  : array-like
    title  : str
    xlabel : str
    color  : str or None
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    counts = np.asarray(counts)
    edges  = np.asarray(edges)
    widths = np.diff(edges)
    c      = color or _COLORS[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        edges[:-1], counts, width=widths,
        align="edge", color=c, edgecolor="white", alpha=0.85,
    )
    _style_ax(ax, title, xlabel, "Count")

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. plot_decision_boundary
# ─────────────────────────────────────────────────────────────────────────────

def plot_decision_boundary(
    model,
    X,
    y,
    feature_indices=(0, 1),
    resolution=150,
    title="Decision Boundary",
    save_path=None,
    figsize=(7, 5),
):
    """
    2-D decision boundary for any classifier with a predict() method.

    Parameters
    ----------
    model : object with predict(X) method
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    feature_indices : tuple (i, j)
        Which two features to use as axes.
    resolution : int
        Grid resolution (higher = smoother but slower).
    title : str
    save_path : str or None
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    i, j    = feature_indices
    padding = 0.5
    x_min, x_max = X[:, i].min() - padding, X[:, i].max() + padding
    y_min, y_max = X[:, j].min() - padding, X[:, j].max() + padding

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # build grid — use column means for non-plotted features
    grid        = np.tile(X.mean(axis=0), (xx.ravel().shape[0], 1))
    grid[:, i]  = xx.ravel()
    grid[:, j]  = yy.ravel()

    classes  = np.unique(y)
    Z        = model.predict(grid)
    Z_idx    = np.searchsorted(classes, Z).reshape(xx.shape)

    fig, ax  = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, Z_idx, alpha=0.35, cmap=plt.cm.Pastel1)

    for k, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(
            X[mask, i], X[mask, j],
            s=20, edgecolors="k", linewidths=0.4,
            color=_COLORS[k % len(_COLORS)],
            label=f"Class {cls}", zorder=2,
        )

    ax.set_xlabel(f"Feature {i}")
    ax.set_ylabel(f"Feature {j}")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=_GRID_ALPHA)

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title, xlabel, ylabel):
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=_GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _set_metric_ylim(ax, ylabel, values):
    """Set y-axis limits to [0, 1.05] for proportion metrics."""
    proportion_names = (
        "accuracy", "precision", "recall", "f1",
        "auc", "score", "metric",
    )
    if any(p in ylabel.lower() for p in proportion_names):
        finite = values[np.isfinite(values)]
        lo     = max(0.0, finite.min() - 0.05) if len(finite) else 0.0
        ax.set_ylim(lo, 1.05)


def _maybe_save(fig, save_path):
    """Save figure to disk if save_path is provided."""
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")