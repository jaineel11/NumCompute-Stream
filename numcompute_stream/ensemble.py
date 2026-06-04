"""
ensemble.py – Tree-based ensemble methods with streaming support.

All methods are exposed through a single EnsembleClassifier class
that switches behaviour via the ``method`` parameter, matching the
spec's requirement for a unified ensemble interface.

Supported methods
-----------------
'bagging'       – Bootstrap Aggregating (Bagging)
'random_forest' – Bagging + random feature subsets per split
'adaboost'      – Adaptive Boosting (AdaBoost-M1)

All three support:
    partial_fit(X_chunk, y_chunk)  – incremental/streaming update
    predict(X)                     – majority vote or weighted vote
    predict_proba(X)               – class probability estimates
    score(X, y)                    – accuracy

Internal implementation uses DecisionTreeClassifier from tree.py
so all NaN safety, criterion options, and stopping criteria are
inherited automatically.
"""

import numpy as np
from .tree import DecisionTreeClassifier


class EnsembleClassifier:
    """
    Unified ensemble classifier supporting Bagging, Random Forest,
    and AdaBoost, all with streaming (partial_fit) support.

    On each partial_fit call, all estimators are rebuilt from the
    full accumulated data buffer using the appropriate sampling
    strategy for the chosen method.

    Parameters
    ----------
    method : str, default='random_forest'
        Ensemble strategy.
        'bagging'       – bootstrap samples, all features per split.
        'random_forest' – bootstrap samples, sqrt(d) features per split.
        'adaboost'      – weighted resampling, shallow trees (stumps).
    n_estimators : int, default=10
        Number of base estimators (trees or boosting rounds).
    max_depth : int, default=5
        Maximum depth of each tree.
        AdaBoost uses max_depth=1 (decision stumps) by default
        unless overridden explicitly.
    max_samples : float, default=1.0
        Fraction of training samples drawn per bootstrap sample.
        Applies to bagging and random_forest only.
    criterion : str, default='gini'
        Split criterion passed to each DecisionTreeClassifier.
        'gini' or 'entropy'.
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    learning_rate : float, default=1.0
        Shrinks each AdaBoost estimator's contribution.
        Applies to adaboost only.
    random_state : int or None, default=None

    Attributes
    ----------
    estimators_        : list[DecisionTreeClassifier]
    estimator_weights_ : np.ndarray
        Per-estimator alpha weights (AdaBoost only; ones elsewhere).
    classes_           : np.ndarray
    n_samples_seen_    : int
    """

    def __init__(
        self,
        method="random_forest",
        n_estimators=10,
        max_depth=5,
        max_samples=1.0,
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        learning_rate=1.0,
        random_state=None,
    ):
        if method not in ("bagging", "random_forest", "adaboost"):
            raise ValueError(
                f"method must be 'bagging', 'random_forest', or "
                f"'adaboost', got '{method}'"
            )
        if not 0 < max_samples <= 1.0:
            raise ValueError(
                f"max_samples must be in (0, 1], got {max_samples}"
            )

        self.method            = method
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.max_samples       = max_samples
        self.criterion         = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.learning_rate     = learning_rate
        self.random_state      = random_state

        self.estimators_        = []
        self.estimator_weights_ = np.array([])
        self.classes_           = None
        self.n_samples_seen_    = 0

        self._rng      = np.random.RandomState(random_state)
        self._X_buffer = None
        self._y_buffer = None

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Fit the ensemble on a full dataset, resetting all prior state.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._X_buffer       = X.copy()
        self._y_buffer       = y.copy()
        self.classes_        = np.unique(y)
        self.n_samples_seen_ = len(y)
        self._build()
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        """
        Incrementally update the ensemble with a new data chunk.

        The data buffer grows with each call and all estimators are
        rebuilt to reflect the full history seen so far.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)
        classes : array-like or None
            All possible classes. Recommended on first call.

        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        if self._X_buffer is None:
            self._X_buffer = X_chunk.copy()
            self._y_buffer = y_chunk.copy()
        else:
            self._X_buffer = np.vstack([self._X_buffer, X_chunk])
            self._y_buffer = np.concatenate([self._y_buffer, y_chunk])

        if classes is not None and self.classes_ is None:
            self.classes_ = np.asarray(classes)

        self.classes_        = np.unique(self._y_buffer)
        self.n_samples_seen_ += len(y_chunk)
        self._build()
        return self

    def predict(self, X):
        """
        Predict class labels.

        Bagging / Random Forest use majority vote.
        AdaBoost uses weighted majority vote.

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
        X         = np.asarray(X, dtype=float)
        n_classes = len(self.classes_)
        n_samples = len(X)

        weighted_votes = np.zeros((n_samples, n_classes))

        for alpha, est in zip(self.estimator_weights_, self.estimators_):
            preds = est.predict(X)
            for i, pred in enumerate(preds):
                j = int(np.searchsorted(self.classes_, pred))
                if j < n_classes and self.classes_[j] == pred:
                    weighted_votes[i, j] += alpha

        return self.classes_[np.argmax(weighted_votes, axis=1)]

    def predict_proba(self, X):
        """
        Return class probability estimates.

        Probabilities are the weighted average of per-estimator
        predict_proba outputs, weighted by estimator_weights_.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)

        Raises
        ------
        RuntimeError : if called before fit / partial_fit.
        """
        self._check_fitted()
        X       = np.asarray(X, dtype=float)
        probas  = np.array([
            est.predict_proba(X) for est in self.estimators_
        ])  # shape (n_estimators, n_samples, n_classes)

        weights = self.estimator_weights_
        weights = weights / weights.sum()
        return np.tensordot(weights, probas, axes=([0], [0]))

    def score(self, X, y):
        """Return accuracy on (X, y)."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))

    # ── internal build dispatch ───────────────────────────────────────────────

    def _build(self):
        """Dispatch to the appropriate build strategy."""
        if self.method in ("bagging", "random_forest"):
            self._build_bagging()
        else:
            self._build_adaboost()

    # ── bagging / random forest ───────────────────────────────────────────────

    def _build_bagging(self):
        """
        Build Bagging or Random Forest ensemble.

        Each estimator trains on a bootstrap sample. Random Forest
        additionally subsamples sqrt(n_features) features per split.
        """
        X, y  = self._X_buffer, self._y_buffer
        n     = len(y)
        k     = max(1, int(n * self.max_samples))
        seeds = self._rng.randint(0, 100_000, size=self.n_estimators)

        if self.method == "random_forest":
            max_features = "sqrt"
        else:
            max_features = None   # bagging uses all features

        self.estimators_ = []
        for seed in seeds:
            rng_local = np.random.RandomState(int(seed))
            idx       = rng_local.choice(n, size=k, replace=True)
            tree      = DecisionTreeClassifier(
                criterion         = self.criterion,
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf  = self.min_samples_leaf,
                max_features      = max_features,
                random_state      = int(seed),
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)

        # uniform weights for voting
        self.estimator_weights_ = np.ones(self.n_estimators)

    # ── adaboost ─────────────────────────────────────────────────────────────

    def _build_adaboost(self):
        """
        Build AdaBoost-M1 ensemble.

        Uses weighted resampling to approximate weighted fitting.
        Each estimator is a decision stump (max_depth=1) unless
        max_depth was explicitly set to something larger.

        Estimator weights (alpha) are proportional to log-odds of
        accuracy — better estimators get higher vote weight.
        """
        X, y = self._X_buffer, self._y_buffer
        n    = len(y)
        w    = np.full(n, 1.0 / n)

        # use stumps by default for AdaBoost
        tree_depth = 1 if self.max_depth == 5 else self.max_depth

        self.estimators_        = []
        alphas                  = []

        for _ in range(self.n_estimators):
            seed      = int(self._rng.randint(0, 100_000))
            rng_local = np.random.RandomState(seed)

            # weighted resample
            idx  = rng_local.choice(n, size=n, replace=True, p=w)
            tree = DecisionTreeClassifier(
                criterion         = self.criterion,
                max_depth         = tree_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf  = self.min_samples_leaf,
                random_state      = seed,
            )
            tree.fit(X[idx], y[idx])

            y_pred    = tree.predict(X)
            incorrect = (y_pred != y).astype(float)
            err       = float(np.dot(w, incorrect))
            err       = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = self.learning_rate * 0.5 * np.log(
                (1.0 - err) / err
            )

            # update sample weights — upweight misclassified
            w *= np.exp(alpha * (2 * incorrect - 1))
            w /= w.sum()

            self.estimators_.append(tree)
            alphas.append(alpha)

        self.estimator_weights_ = np.array(alphas)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.estimators_:
            raise RuntimeError(
                "EnsembleClassifier has not been fitted yet. "
                "Call fit() or partial_fit() first."
            )