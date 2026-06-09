"""
stats.py – Streaming descriptive statistics.

All classes use Welford's online algorithm for numerical stability.
Every class exposes a consistent API:

    update(data)   → ingest a new chunk, update running state
    compute()      → return current statistic(s) as dict or tuple
    reset()        → clear all accumulated state

The module-level update_stats(X_chunk) convenience function mirrors
the spec's required API for quick one-liner stat collection.

Classes
-------
StreamingStats      – mean, variance, std, min, max over 1-D or 2-D chunks
StreamingHistogram  – fixed-bin histogram with optional sliding window
ExponentialMovingAverage – EWMA of a scalar signal
"""

import numpy as np
from collections import deque


# Module-level convenience function  (spec: update_stats API)

# A single module-level instance so callers can do:
#   from numcompute_stream.stats import update_stats
#   stats = update_stats(X_chunk)
_global_stats = None


def update_stats(X_chunk):
    """
    Update a module-level StreamingStats instance with a new chunk.

    Provides the spec's required ``update_stats(X_chunk)`` API for
    quick per-chunk descriptive statistics without managing an object.

    Parameters
    ----------
    X_chunk : array-like, shape (n_samples,) or (n_samples, n_features)

    Returns
    -------
    dict with keys: mean, var, std, min, max, n

    Example
    -------
    >>> for chunk in chunks:
    ...     stats = update_stats(chunk)
    ...     print(stats["mean"])
    """
    global _global_stats
    if _global_stats is None:
        _global_stats = StreamingStats()
    _global_stats.update(X_chunk)
    return _global_stats.compute()


def reset_stats():
    """Reset the module-level StreamingStats instance."""
    global _global_stats
    _global_stats = None


# StreamingStats

class StreamingStats:
    """
    Maintain running mean, variance, min, max, and count via Welford's
    online algorithm.

    Handles 1-D or 2-D input:
        1-D : statistics over all elements treated as a single feature.
        2-D : per-column (feature) statistics.

    NaN values are silently skipped — the running mean is substituted
    before each Welford update so statistics remain unbiased.

    Parameters
    ----------
    ddof : int, default=0
        Degrees of freedom for variance.
        0 → population variance, 1 → sample variance.

    Attributes
    ----------
    mean_ : float or np.ndarray
    var_  : float or np.ndarray  (property)
    std_  : float or np.ndarray  (property)
    min_  : float or np.ndarray
    max_  : float or np.ndarray
    n_samples_seen_ : int
    """

    def __init__(self, ddof=0):
        self.ddof             = ddof
        self.mean_            = None
        self._M2              = None
        self.min_             = None
        self.max_             = None
        self.n_samples_seen_  = 0
        self._is_2d           = None

    # ── public API ───────────────────────────────────────────────────────────

    def update(self, data):
        """
        Incrementally update statistics with new data.

        Parameters
        ----------
        data : array-like, shape (n,) or (n, d)
            New observations. NaN values are handled safely.

        Returns
        -------
        self
        """
        data = np.asarray(data, dtype=float)
        if data.ndim > 2:
            raise ValueError(
                f"data must be 1-D or 2-D, got shape {data.shape}"
            )

        if self._is_2d is None:
            self._is_2d  = (data.ndim == 2)
            n_feats      = data.shape[1] if self._is_2d else 1
            self.mean_   = np.zeros(n_feats)
            self._M2     = np.zeros(n_feats)
            self.min_    = np.full(n_feats,  np.inf)
            self.max_    = np.full(n_feats, -np.inf)

        rows = data if self._is_2d else data.reshape(-1, 1)

        for row in rows:
            nan_mask = np.isnan(row)
            safe_row = np.where(nan_mask, self.mean_, row)
            self.n_samples_seen_ += 1
            delta      = safe_row - self.mean_
            self.mean_ += delta / self.n_samples_seen_
            delta2     = safe_row - self.mean_
            self._M2   += delta * delta2
            non_nan     = ~nan_mask
            self.min_   = np.where(non_nan, np.minimum(self.min_, row), self.min_)
            self.max_   = np.where(non_nan, np.maximum(self.max_, row), self.max_)

        return self

    def compute(self):
        """
        Return all current statistics as a dict.

        Returns
        -------
        dict with keys: mean, var, std, min, max, n
        """
        return {
            "mean": self._out(self.mean_),
            "var":  self.var_,
            "std":  self.std_,
            "min":  self._out(self.min_),
            "max":  self._out(self.max_),
            "n":    self.n_samples_seen_,
        }

    def reset(self):
        """Clear all accumulated state."""
        self.mean_           = None
        self._M2             = None
        self.min_            = None
        self.max_            = None
        self.n_samples_seen_ = 0
        self._is_2d          = None
        return self

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def var_(self):
        if self._M2 is None:
            return None
        denom = max(1, self.n_samples_seen_ - self.ddof)
        return self._out(self._M2 / denom)

    @property
    def std_(self):
        v = self.var_
        if v is None:
            return None
        return float(np.sqrt(v)) if np.isscalar(v) else np.sqrt(v)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _out(self, arr):
        """Squeeze to scalar when input was 1-D."""
        if arr is None:
            return None
        if not self._is_2d and arr is not None:
            arr = np.asarray(arr)
            return float(arr[0]) if arr.size == 1 else arr
        return arr


# StreamingHistogram

class StreamingHistogram:
    """
    Accumulate a histogram from streaming 1-D data with fixed bins.

    Optionally maintains a sliding window so older data is dropped
    and the histogram reflects only the most recent observations.

    Parameters
    ----------
    bins : int or array-like, default=20
        Number of equal-width bins, or explicit bin edges.
    range_ : tuple (min, max) or None, default=None
        Required when bins is an int. Defines the fixed bin edges.
    window : int or None, default=None
        If set, only the most recent ``window`` observations contribute
        to the histogram (sliding window mode).

    Attributes
    ----------
    counts_ : np.ndarray
        Accumulated bin counts.
    edges_  : np.ndarray
        Bin edges (length = len(counts_) + 1).
    """

    def __init__(self, bins=20, range_=None, window=None):
        self.bins         = bins
        self.range_       = range_
        self.window       = window
        self.counts_      = None
        self.edges_       = None
        self._initialized = False
        # sliding window buffer
        self._buffer      = deque(maxlen=window) if window else None

    def update(self, data):
        """
        Add new 1-D observations to the histogram.

        Parameters
        ----------
        data : array-like, shape (n,)
            NaN values are silently dropped.

        Returns
        -------
        self
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise ValueError(
                f"data must be 1-D for histogram, got shape {data.shape}"
            )
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return self

        if self.window is not None:
            # sliding window — buffer new values, rebuild histogram each time
            self._buffer.extend(data.tolist())
            data_to_bin = np.asarray(self._buffer)
        else:
            data_to_bin = data

        if not self._initialized:
            r = self.range_
            if r is None:
                r = (data_to_bin.min(), data_to_bin.max())
            if r[0] == r[1]:
                r = (r[0] - 1, r[0] + 1)
            counts, self.edges_ = np.histogram(
                data_to_bin, bins=self.bins, range=r
            )
            self.counts_      = counts
            self._initialized = True
        else:
            if self.window is not None:
                # full rebuild from window buffer
                counts, _ = np.histogram(data_to_bin, bins=self.edges_)
                self.counts_ = counts
            else:
                counts, _ = np.histogram(data, bins=self.edges_)
                self.counts_ += counts

        return self

    def compute(self):
        """
        Return current histogram state.

        Returns
        -------
        counts : np.ndarray
        edges  : np.ndarray
        """
        if self.counts_ is None:
            return np.array([]), np.array([])
        return self.counts_.copy(), self.edges_.copy()

    def reset(self):
        """Clear accumulated histogram state."""
        self.counts_      = None
        self.edges_       = None
        self._initialized = False
        if self.window:
            self._buffer  = deque(maxlen=self.window)
        return self


# ExponentialMovingAverage

class ExponentialMovingAverage:
    """
    Exponentially weighted moving average (EWMA) of a scalar signal.

    Useful for smoothing noisy per-chunk metrics (e.g., accuracy over time).

    Parameters
    ----------
    alpha : float, default=0.1
        Smoothing factor in (0, 1].
        Higher alpha → more weight on recent observations.

    Attributes
    ----------
    value_   : float or None
        Current EWMA value.
    history_ : list[float]
        EWMA value after each update call.
    """

    def __init__(self, alpha=0.1):
        if not 0 < alpha <= 1:
            raise ValueError(
                f"alpha must be in (0, 1], got {alpha}"
            )
        self.alpha   = alpha
        self.value_  = None
        self.history_ = []

    def update(self, x):
        """
        Update the EWMA with a new scalar observation.

        NaN values are silently ignored (value_ unchanged).

        Parameters
        ----------
        x : float

        Returns
        -------
        self
        """
        x = float(x)
        if np.isnan(x):
            return self
        if self.value_ is None:
            self.value_ = x
        else:
            self.value_ = self.alpha * x + (1 - self.alpha) * self.value_
        self.history_.append(self.value_)
        return self

    def compute(self):
        """Return the current EWMA value (None if no data seen)."""
        return self.value_

    def reset(self):
        """Clear accumulated state."""
        self.value_   = None
        self.history_ = []
        return self