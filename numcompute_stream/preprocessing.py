"""
preprocessing.py – Streaming-compatible preprocessors.

All classes implement:
    fit(X)           → fits on full dataset, resets state
    partial_fit(X)   → incremental update (Welford for scalers)
    transform(X)     → apply learned transform
    fit_transform(X) → fit + transform in one step

Classes
-------
StandardScaler    – zero-mean, unit-variance via Welford online algorithm
MinMaxScaler      – scale to feature_range via running min/max
Imputer           – fill missing values with running mean/median/constant
OneHotEncoder     – encode categoricals, expands columns incrementally
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# StandardScaler
# ─────────────────────────────────────────────────────────────────────────────

class StandardScaler:
    """
    Standardise features to zero mean and unit variance.

    Uses Welford's online algorithm for numerical stability — safe for
    streaming where data arrives chunk by chunk.

    NaN inputs: ignored during stats update, replaced with 0.0 in output.

    Parameters
    ----------
    None

    Attributes
    ----------
    mean_ : np.ndarray, shape (n_features,)
    var_  : np.ndarray, shape (n_features,)
    std_  : np.ndarray, shape (n_features,)
    n_samples_seen_ : int
    """

    def __init__(self):
        self.mean_ = None
        self.var_  = None
        self.std_  = None
        self.n_samples_seen_ = 0
        self._M2 = None

    def fit(self, X):
        """
        Fit on a full dataset, resetting all prior state.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        self.n_samples_seen_ = 0
        self.mean_ = np.zeros(X.shape[1])
        self._M2   = np.zeros(X.shape[1])
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Incrementally update running statistics with a new chunk.

        Uses Welford's single-pass algorithm — numerically stable for
        large or shifted values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        n_features = X.shape[1]

        if self.mean_ is None:
            self.mean_ = np.zeros(n_features)
            self._M2   = np.zeros(n_features)

        for row in X:
            nan_mask = np.isnan(row)
            safe_row = np.where(nan_mask, self.mean_, row)
            self.n_samples_seen_ += 1
            delta  = safe_row - self.mean_
            self.mean_ += delta / self.n_samples_seen_
            delta2 = safe_row - self.mean_
            self._M2 += delta * delta2

        if self.n_samples_seen_ > 1:
            self.var_ = self._M2 / (self.n_samples_seen_ - 1)
        else:
            self.var_ = np.zeros(n_features)

        self.std_ = np.sqrt(self.var_)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        """
        Standardise X using current running mean and std.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            NaN positions become 0.0 after standardisation.

        Raises
        ------
        RuntimeError : if called before fit/partial_fit.
        """
        if self.mean_ is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        X = self._validate(X)
        nan_mask = np.isnan(X)
        Xt = (X - self.mean_) / self.std_
        Xt[nan_mask] = 0.0
        return Xt

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _validate(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D, got shape {X.shape}. "
                "Reshape with X.reshape(-1, 1) for a single feature."
            )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# MinMaxScaler
# ─────────────────────────────────────────────────────────────────────────────

class MinMaxScaler:
    """
    Scale each feature to a given range using running min/max.

    On partial_fit the running min and max are updated so any new
    chunk that extends the observed range is handled correctly.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)

    Attributes
    ----------
    min_ : np.ndarray
    max_ : np.ndarray
    n_samples_seen_ : int
    """

    def __init__(self, feature_range=(0, 1)):
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                f"feature_range min must be < max, got {feature_range}"
            )
        self.feature_range    = feature_range
        self.min_             = None
        self.max_             = None
        self.n_samples_seen_  = 0

    def fit(self, X):
        X = self._validate(X)
        self.min_ = None
        self.max_ = None
        self.n_samples_seen_ = 0
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Update running min/max with a new chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)

        if self.min_ is None:
            self.min_ = chunk_min.copy()
            self.max_ = chunk_max.copy()
        else:
            self.min_ = np.minimum(self.min_, chunk_min)
            self.max_ = np.maximum(self.max_, chunk_max)

        self.n_samples_seen_ += len(X)
        return self

    def transform(self, X):
        """
        Scale X into feature_range using the running min/max.

        Raises
        ------
        RuntimeError : if called before fit/partial_fit.
        """
        if self.min_ is None:
            raise RuntimeError("MinMaxScaler has not been fitted yet.")
        X = self._validate(X)
        scale = self.max_ - self.min_
        scale[scale == 0] = 1.0
        a, b = self.feature_range
        return (X - self.min_) / scale * (b - a) + a

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _validate(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Imputer
# ─────────────────────────────────────────────────────────────────────────────

class Imputer:
    """
    Fill missing values (NaN) using a streaming-compatible strategy.

    The fill value is updated incrementally on each partial_fit call,
    so estimates improve as more data is seen.

    Parameters
    ----------
    strategy : str, default='mean'
        'mean'     – replace with running per-column mean (Welford).
        'median'   – replace with per-column median of all seen data
                     (exact; buffers column values).
        'constant' – replace with fill_value.
    fill_value : scalar, default=0.0
        Used only when strategy='constant'.

    Attributes
    ----------
    statistics_ : np.ndarray, shape (n_features,)
        The fill value per feature after fitting.
    n_samples_seen_ : int
    """

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in ("mean", "median", "constant"):
            raise ValueError(
                f"strategy must be 'mean', 'median', or 'constant', got '{strategy}'"
            )
        self.strategy        = strategy
        self.fill_value      = fill_value
        self.statistics_     = None
        self.n_samples_seen_ = 0
        # internal Welford state for mean
        self._mean           = None
        self._M2             = None
        # buffer for median (stores non-NaN values per column)
        self._col_buffers    = None

    def fit(self, X):
        """Fit on full dataset, resetting prior state."""
        X = self._validate(X)
        self.statistics_     = None
        self.n_samples_seen_ = 0
        self._mean           = None
        self._M2             = None
        self._col_buffers    = None
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Incrementally update fill-value estimates with a new chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        n_features = X.shape[1]

        if self.strategy == "constant":
            self.statistics_ = np.full(n_features, self.fill_value)
            self.n_samples_seen_ += len(X)
            return self

        if self.strategy == "mean":
            if self._mean is None:
                self._mean        = np.zeros(n_features)
                self._M2          = np.zeros(n_features)
                self._col_counts  = np.zeros(n_features)

            for row in X:
                nan_mask = np.isnan(row)
                for j in range(n_features):
                    if nan_mask[j]:
                        continue
                    self._col_counts[j] += 1
                    delta = row[j] - self._mean[j]
                    self._mean[j] += delta / self._col_counts[j]

            self.n_samples_seen_ += len(X)
            self.statistics_ = self._mean.copy()

        elif self.strategy == "median":
            if self._col_buffers is None:
                self._col_buffers = [[] for _ in range(n_features)]

            for col_idx in range(n_features):
                col = X[:, col_idx]
                valid = col[~np.isnan(col)]
                self._col_buffers[col_idx].extend(valid.tolist())

            self.n_samples_seen_ += len(X)
            self.statistics_ = np.array([
                np.median(buf) if buf else 0.0
                for buf in self._col_buffers
            ])

        return self

    def transform(self, X):
        """
        Fill NaN values in X using the learned statistics.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)

        Raises
        ------
        RuntimeError : if called before fit/partial_fit.
        """
        if self.statistics_ is None:
            raise RuntimeError("Imputer has not been fitted yet.")
        X = self._validate(X).copy()
        nan_mask = np.isnan(X)
        # broadcast fill: take column statistic for each NaN position
        fill = np.take(self.statistics_, np.where(nan_mask)[1])
        X[nan_mask] = fill
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _validate(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        return X


# ─────────────────────────────────────────────────────────────────────────────
# OneHotEncoder
# ─────────────────────────────────────────────────────────────────────────────

class OneHotEncoder:
    """
    Encode integer or string categorical columns as one-hot vectors.

    Supports incremental category discovery via partial_fit — new
    categories seen in later chunks are added to the encoding, expanding
    the output width automatically.

    Parameters
    ----------
    sparse : bool, default=False
        If True, output is a list of (row, col) index arrays instead of
        a dense array. Currently always False (dense only).

    Attributes
    ----------
    categories_ : list[np.ndarray]
        Per-feature sorted unique categories seen during fitting.
    n_features_in_ : int
    """

    def __init__(self):
        self.categories_    = None
        self.n_features_in_ = None

    def fit(self, X):
        """Fit on full dataset, resetting prior state."""
        X = self._validate(X)
        self.categories_    = None
        self.n_features_in_ = None
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Discover new categories incrementally.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Each column is treated as a categorical feature.

        Returns
        -------
        self
        """
        X = self._validate(X)
        n_features = X.shape[1]

        if self.categories_ is None:
            self.n_features_in_ = n_features
            self.categories_ = [np.unique(X[:, j]) for j in range(n_features)]
        else:
            if n_features != self.n_features_in_:
                raise ValueError(
                    f"X has {n_features} features but encoder was fitted "
                    f"on {self.n_features_in_}."
                )
            for j in range(n_features):
                self.categories_[j] = np.union1d(
                    self.categories_[j], np.unique(X[:, j])
                )
        return self

    def transform(self, X):
        """
        One-hot encode X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, sum(len(cats) for cats in categories_))

        Raises
        ------
        RuntimeError : if called before fit/partial_fit.
        ValueError   : if an unknown category is encountered.
        """
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder has not been fitted yet.")
        X = self._validate(X)
        n_samples = X.shape[0]
        parts = []

        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            # searchsorted gives index into cats for each value
            idx = np.searchsorted(cats, col)
            # bounds check — catch unknown categories
            # must check idx < len(cats) first before indexing into cats
            in_bounds = idx < len(cats)
            valid = in_bounds & (cats[np.minimum(idx, len(cats) - 1)] == col)
            if not np.all(valid):
                bad = np.unique(col[~valid])
                raise ValueError(
                    f"Feature {j} contains unknown categories: {bad.tolist()}"
                )
            ohe = np.zeros((n_samples, len(cats)), dtype=float)
            ohe[np.arange(n_samples), idx] = 1.0
            parts.append(ohe)

        return np.hstack(parts)

    def inverse_transform(self, X):
        """
        Recover original categories from one-hot encoded array.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, total_categories)

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
        """
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder has not been fitted yet.")
        X = np.asarray(X)
        result = []
        col = 0
        for cats in self.categories_:
            block = X[:, col: col + len(cats)]
            result.append(cats[np.argmax(block, axis=1)])
            col += len(cats)
        return np.column_stack(result)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _validate(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        return X