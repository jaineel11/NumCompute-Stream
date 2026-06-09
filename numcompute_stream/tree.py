"""
tree.py – Streaming-compatible Decision Tree for classification.

Implements DecisionTreeClassifier with:
    - Gini impurity or entropy information gain for splits
    - depth-limited growth via max_depth
    - min_samples_split and min_samples_leaf stopping criteria
    - random feature subsampling via max_features
    - partial_fit() for incremental/online learning
    - NaN-safe input validation (column-median imputation)

The tree is rebuilt from scratch on each partial_fit call using
all accumulated data. This guarantees exact optimal splits at
every stage while keeping the API stateless and simple.
"""

import numpy as np


# Internal node structure

class _Node:
    """A single node in the decision tree — internal split or leaf."""

    __slots__ = [
        "feature", "threshold", "left", "right",
        "is_leaf", "class_counts", "n_samples", "prediction",
    ]

    def __init__(self):
        self.feature      = None
        self.threshold    = None
        self.left         = None
        self.right        = None
        self.is_leaf      = False
        self.class_counts = None   # dict {class_label: count}
        self.n_samples    = 0
        self.prediction   = None   # majority class at this node


# DecisionTreeClassifier

class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier with streaming support.

    Supports both batch (fit) and incremental (partial_fit) training.
    On each partial_fit call the tree is rebuilt from all accumulated
    data, guaranteeing exact Gini or entropy splits at every stage.

    Parameters
    ----------
    criterion : str, default='gini'
        Split quality measure.
        'gini'    → Gini impurity.
        'entropy' → Information gain (Shannon entropy).
    max_depth : int or None, default=5
        Maximum depth of the tree.
        None means nodes expand until other stopping criteria are met.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at each leaf node.
    max_features : int, float, str, or None, default=None
        Number of features to consider per split.
        None   → all features.
        'sqrt' → int(sqrt(n_features)).
        'log2' → int(log2(n_features)).
        float  → fraction of features.
        int    → exact count.
    random_state : int or None, default=None
        Seed for reproducibility when max_features subsamples.

    Attributes
    ----------
    root_            : _Node
    classes_         : np.ndarray
    n_features_      : int
    n_samples_seen_  : int
        Total samples seen across all partial_fit calls.
    """

    def __init__(
        self,
        criterion="gini",
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got '{criterion}'"
            )
        self.criterion        = criterion
        self.max_depth        = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features     = max_features
        self.random_state     = random_state

        self.root_           = None
        self.classes_        = None
        self.n_features_     = None
        self.n_samples_seen_ = 0

        self._X_buffer = None
        self._y_buffer = None
        self._rng      = np.random.RandomState(random_state)

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Fit the tree on a full dataset, replacing any prior state.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = self._validate(X, y)
        self._X_buffer       = X.copy()
        self._y_buffer       = y.copy()
        self.classes_        = np.unique(y)
        self.n_features_     = X.shape[1]
        self.n_samples_seen_ = len(y)
        self.root_           = self._build(X, y, depth=0)
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """
        Incrementally update the tree with a new data chunk.

        All previously seen data is retained in a buffer and the
        tree is rebuilt from scratch to guarantee optimal splits.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)
        classes : array-like or None
            All possible classes. Recommended on first call so the
            tree knows the full label space from the start.

        Returns
        -------
        self
        """
        X_chunk, y_chunk = self._validate(X_chunk, y_chunk)

        # accumulate data
        if self._X_buffer is None:
            self._X_buffer = X_chunk.copy()
            self._y_buffer = y_chunk.copy()
        else:
            self._X_buffer = np.vstack([self._X_buffer, X_chunk])
            self._y_buffer = np.concatenate([self._y_buffer, y_chunk])

        # merge known classes
        if classes is not None:
            if self.classes_ is None:
                self.classes_ = np.asarray(classes)
            else:
                self.classes_ = np.union1d(self.classes_, classes)

        self.classes_        = np.unique(self._y_buffer) \
                               if self.classes_ is None else self.classes_
        self.n_features_     = self._X_buffer.shape[1]
        self.n_samples_seen_ += len(y_chunk)
        self.root_           = self._build(
            self._X_buffer, self._y_buffer, depth=0
        )
        return self

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)

        Raises
        ------
        RuntimeError : if called before fit / partial_fit.
        """
        self._check_fitted()
        X = self._validate_X(X)
        return np.array([
            self._traverse(row, self.root_).prediction
            for row in X
        ])

    def predict_proba(self, X):
        """
        Return class probability estimates.

        Probabilities are the fraction of training samples of each
        class at the leaf node the sample reaches.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        X        = self._validate_X(X)
        n_cls    = len(self.classes_)
        proba    = np.zeros((len(X), n_cls))

        for i, row in enumerate(X):
            node  = self._traverse(row, self.root_)
            total = max(node.n_samples, 1)
            for j, cls in enumerate(self.classes_):
                proba[i, j] = node.class_counts.get(cls, 0) / total

        return proba

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))

    def get_depth(self):
        """Return the actual depth of the fitted tree."""
        def _d(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_d(node.left), _d(node.right))
        return _d(self.root_)

    # ── impurity criteria ────────────────────────────────────────────────────

    def _impurity(self, y):
        """
        Compute impurity of label array y.

        Returns Gini impurity or Shannon entropy depending on
        self.criterion. Returns 0.0 for empty arrays.
        """
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs     = counts / len(y)

        if self.criterion == "gini":
            return float(1.0 - np.sum(probs ** 2))
        else:  # entropy
            # clip to avoid log(0)
            probs  = np.clip(probs, 1e-12, 1.0)
            return float(-np.sum(probs * np.log2(probs)))

    # ── split search ─────────────────────────────────────────────────────────

    def _best_split(self, X, y):
        """
        Find the best (feature, threshold) split.

        Evaluates midpoints between consecutive unique values for each
        (sub-sampled) feature. Selects the split that maximises
        impurity reduction (information gain).

        Returns
        -------
        best_feat      : int or None
        best_threshold : float or None
        """
        n_samples, n_features = X.shape
        best_gain  = -np.inf
        best_feat  = None
        best_thresh = None
        current_imp = self._impurity(y)

        n_feat_use   = self._n_features_to_use(n_features)
        feat_indices = self._rng.choice(
            n_features, n_feat_use, replace=False
        )

        for feat in feat_indices:
            col        = X[:, feat]
            thresholds = np.unique(col)
            if len(thresholds) <= 1:
                continue
            # midpoints between consecutive unique values
            mids = (thresholds[:-1] + thresholds[1:]) / 2.0

            for thresh in mids:
                left_mask  = col <= thresh
                right_mask = ~left_mask
                n_left     = left_mask.sum()
                n_right    = right_mask.sum()

                if (n_left  < self.min_samples_leaf or
                        n_right < self.min_samples_leaf):
                    continue

                imp_left   = self._impurity(y[left_mask])
                imp_right  = self._impurity(y[right_mask])
                weighted   = (
                    n_left  * imp_left +
                    n_right * imp_right
                ) / n_samples
                gain = current_imp - weighted

                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _n_features_to_use(self, n_features):
        """Resolve max_features to an integer count."""
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(max(n_features, 2))))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        return min(int(mf), n_features)

    # ── tree construction ────────────────────────────────────────────────────

    def _build(self, X, y, depth):
        """Recursively build the tree, returning a _Node."""
        node           = _Node()
        node.n_samples = len(y)
        unique, counts = np.unique(y, return_counts=True)
        node.class_counts = dict(zip(unique.tolist(), counts.tolist()))
        node.prediction   = unique[np.argmax(counts)]

        # stopping criteria
        at_max_depth = (
            self.max_depth is not None and depth >= self.max_depth
        )
        too_small = len(y) < self.min_samples_split
        pure      = len(unique) == 1

        if at_max_depth or too_small or pure:
            node.is_leaf = True
            return node

        feat, thresh = self._best_split(X, y)
        if feat is None:
            node.is_leaf = True
            return node

        node.feature   = feat
        node.threshold = thresh
        left_mask      = X[:, feat] <= thresh
        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    # ── traversal ────────────────────────────────────────────────────────────

    def _traverse(self, row, node):
        """Walk the tree for a single sample, returning its leaf node."""
        if node.is_leaf:
            return node
        if row[node.feature] <= node.threshold:
            return self._traverse(row, node.left)
        return self._traverse(row, node.right)

    # ── validation ───────────────────────────────────────────────────────────

    def _validate(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-D, got shape {X.shape}."
            )
        if y.ndim != 1:
            raise ValueError(
                f"y must be 1-D, got shape {y.shape}."
            )
        if len(X) != len(y):
            raise ValueError(
                f"X and y length mismatch: {len(X)} vs {len(y)}."
            )
        # NaN imputation — replace with column median
        col_medians = np.nanmedian(X, axis=0)
        nan_mask    = np.isnan(X)
        X           = X.copy()
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        return X, y

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        col_medians = np.nanmedian(X, axis=0)
        nan_mask    = np.isnan(X)
        X           = X.copy()
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        return X

    def _check_fitted(self):
        if self.root_ is None:
            raise RuntimeError(
                "DecisionTreeClassifier has not been fitted yet. "
                "Call fit() or partial_fit() first."
            )