"""
pipeline.py – Streaming ML pipeline.

Pipeline chains an ordered sequence of transformers followed by a
final estimator. Every step supports partial_fit() so the whole
pipeline can be updated incrementally as new data chunks arrive.

Usage
-----
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', EnsembleClassifier(method='random_forest')),
    ])

    for X_chunk, y_chunk in chunks:
        pipe.partial_fit(X_chunk, y_chunk)

    predictions = pipe.predict(X_test)
"""

import numpy as np


class Pipeline:
    """
    Chain transformers and a final estimator, all supporting partial_fit.

    Steps are applied in order during both fit and transform. On
    partial_fit, every transformer is updated incrementally before the
    transformed chunk is passed to the estimator.

    Parameters
    ----------
    steps : list of (name, estimator) tuples
        Ordered list of (name, object) pairs. All steps except the
        last must implement partial_fit(X) or fit(X) and transform(X).
        The last step must implement partial_fit(X, y) or fit(X, y)
        and predict(X).

    Attributes
    ----------
    steps        : list of (name, estimator) tuples
    transformers_: list
        All steps except the last.
    estimator_   : object
        The final estimator step.
    n_samples_seen_ : int
        Total samples seen across all partial_fit calls.
    metric_log_  : list[dict]
        Per-chunk log entries when X_eval / y_eval are supplied to
        partial_fit. Each entry has keys: chunk, n_samples, accuracy.
    """

    def __init__(self, steps):
        if len(steps) == 0:
            raise ValueError(
                "steps cannot be empty. Provide at least one estimator."
            )
        for name, obj in steps:
            if not isinstance(name, str):
                raise TypeError(
                    f"Step names must be strings, got {type(name)}."
                )
            if obj is None:
                raise ValueError(
                    f"Step '{name}' is None. All steps must be estimator objects."
                )

        self.steps        = steps
        self.transformers_ = [obj for _, obj in steps[:-1]]
        self.estimator_   = steps[-1][1]
        self.n_samples_seen_ = 0
        self.metric_log_  = []
        self._chunk_count = 0

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Fit all steps on a full dataset.

        Each transformer is fit_transformed in sequence; the final
        estimator is fit on the fully transformed output.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X  = np.asarray(X, dtype=float)
        y  = np.asarray(y)
        Xt = self._fit_transform_all(X)
        self.estimator_.fit(Xt, y)
        self.n_samples_seen_ = len(y)
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None,
                    X_eval=None, y_eval=None):
        """
        Incrementally update all pipeline steps with a new chunk.

        Each transformer is updated via partial_fit(X) then used to
        transform the chunk before passing it downstream. The final
        estimator receives the fully transformed chunk.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)
        classes : array-like or None
            All possible classes. Forwarded to the estimator's
            partial_fit if it accepts a classes argument.
        X_eval  : array-like or None
            Optional held-out data for per-chunk accuracy logging.
        y_eval  : array-like or None
            Labels for X_eval.

        Returns
        -------
        self
        """
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        Xt = self._partial_fit_transform_all(X_chunk)

        # update estimator
        est = self.estimator_
        if hasattr(est, "partial_fit"):
            try:
                est.partial_fit(Xt, y_chunk, classes=classes)
            except TypeError:
                # estimator doesn't accept classes kwarg
                est.partial_fit(Xt, y_chunk)
        else:
            est.fit(Xt, y_chunk)

        self.n_samples_seen_ += len(y_chunk)
        self._chunk_count    += 1

        # optional per-chunk metric logging
        if X_eval is not None and y_eval is not None:
            acc = self.score(X_eval, y_eval)
            self.metric_log_.append({
                "chunk":     self._chunk_count,
                "n_samples": self.n_samples_seen_,
                "accuracy":  acc,
            })

        return self

    def predict(self, X):
        """
        Transform X through all transformers then predict.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        X  = np.asarray(X, dtype=float)
        Xt = self._transform_all(X)
        return self.estimator_.predict(Xt)

    def predict_proba(self, X):
        """
        Transform X then return class probability estimates.

        The final estimator must implement predict_proba.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)

        Raises
        ------
        AttributeError : if the estimator does not support predict_proba.
        """
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError(
                f"{type(self.estimator_).__name__} does not support "
                "predict_proba."
            )
        X  = np.asarray(X, dtype=float)
        Xt = self._transform_all(X)
        return self.estimator_.predict_proba(Xt)

    def score(self, X, y):
        """
        Return accuracy on (X, y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        float
        """
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))

    def get_metric_history(self, metric="accuracy"):
        """
        Return list of a logged metric across all partial_fit calls.

        Parameters
        ----------
        metric : str, default='accuracy'
            Key in the metric_log_ dicts.

        Returns
        -------
        list[float]
        """
        return [
            entry[metric]
            for entry in self.metric_log_
            if metric in entry
        ]

    def get_params(self):
        """
        Return a dict of step names to estimator objects.

        Returns
        -------
        dict {name: estimator}
        """
        return {name: obj for name, obj in self.steps}

    # ── internal transform helpers ────────────────────────────────────────────

    def _fit_transform_all(self, X):
        """fit_transform every transformer in sequence."""
        Xt = X
        for t in self.transformers_:
            if hasattr(t, "fit_transform"):
                Xt = t.fit_transform(Xt)
            else:
                t.fit(Xt)
                Xt = t.transform(Xt)
        return Xt

    def _partial_fit_transform_all(self, X):
        """partial_fit then transform every transformer in sequence."""
        Xt = X
        for t in self.transformers_:
            if hasattr(t, "partial_fit"):
                t.partial_fit(Xt)
                Xt = t.transform(Xt)
            elif hasattr(t, "fit_transform"):
                Xt = t.fit_transform(Xt)
            else:
                t.fit(Xt)
                Xt = t.transform(Xt)
        return Xt

    def _transform_all(self, X):
        """Apply transform on every transformer in sequence."""
        Xt = X
        for t in self.transformers_:
            Xt = t.transform(Xt)
        return Xt

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self):
        step_strs = "\n  ".join(
            f"({name!r}, {type(obj).__name__})"
            for name, obj in self.steps
        )
        return f"Pipeline([\n  {step_strs}\n])"