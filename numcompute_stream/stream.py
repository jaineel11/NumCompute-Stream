"""
stream.py – StreamTrainer: central controller for streaming ML workflows.

StreamTrainer wraps a Pipeline (or any estimator with partial_fit)
and adds:
    - fit_chunk(X, y)        incremental training on one chunk
    - score_chunk(X, y)      evaluate on a held-out set after each chunk
    - per-chunk metric log   accuracy, loss, memory footprint, elapsed time
    - cumulative accuracy    running accuracy across all chunks seen
    - summary()              human-readable training log table
    - log_to_csv()           persist the log to disk for later analysis

All logging is pure Python + NumPy — no external dependencies.

Usage
-----
    from numcompute_stream import Pipeline, StandardScaler, EnsembleClassifier
    from numcompute_stream.stream import StreamTrainer
    from numcompute_stream.io import split_into_chunks, train_test_split

    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', EnsembleClassifier(method='random_forest', n_estimators=10)),
    ])

    trainer = StreamTrainer(pipeline=pipe, verbose=True)

    for X_chunk, y_chunk in split_into_chunks(X_train, y_train, n_chunks=10):
        trainer.fit_chunk(X_chunk, y_chunk)
        print(trainer.score_chunk(X_test, y_test))

    trainer.summary()
    trainer.log_to_csv('logs/training_log.csv')
"""

import time
import sys
import numpy as np
from pathlib import Path


class StreamTrainer:
    """
    Central controller for streaming / incremental ML workflows.

    Manages a pipeline or estimator, drives chunk-by-chunk training,
    and maintains a detailed per-chunk log covering accuracy, memory
    footprint, elapsed time, and cumulative accuracy.

    Parameters
    ----------
    pipeline : object
        Any object with partial_fit(X, y) and predict(X).
        Typically a Pipeline instance but can be a bare estimator.
    classes : array-like or None, default=None
        All possible class labels. If None, inferred from first chunk.
    verbose : bool, default=False
        If True, print a one-line summary after each fit_chunk call.

    Attributes
    ----------
    log_ : list[dict]
        One entry per fit_chunk call. Keys:
            chunk          – 1-based chunk index
            n_samples      – samples in this chunk
            n_total        – cumulative samples seen
            train_acc      – accuracy on the training chunk just fitted
            eval_acc       – accuracy on eval set (None if not provided)
            cumulative_acc – running accuracy across all eval calls
            memory_bytes   – estimated memory of the chunk (X + y)
            elapsed_s      – wall-clock seconds for fit_chunk
    n_chunks_seen_ : int
    n_samples_seen_ : int
    classes_ : np.ndarray or None
    """

    def __init__(self, pipeline, classes=None, verbose=False):
        self.pipeline  = pipeline
        self.classes_  = np.asarray(classes) if classes is not None else None
        self.verbose   = verbose

        self.log_            = []
        self.n_chunks_seen_  = 0
        self.n_samples_seen_ = 0

        # internal accumulators for cumulative accuracy
        self._cum_correct = 0
        self._cum_total   = 0

    # ── core API ─────────────────────────────────────────────────────────────

    def fit_chunk(self, X, y):
        """
        Train the pipeline on one chunk of data.

        Calls pipeline.partial_fit(X, y) and records timing and
        memory stats. Optionally prints progress if verbose=True.

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

        # infer classes from first chunk if not provided
        if self.classes_ is None:
            self.classes_ = np.unique(y)

        t0 = time.perf_counter()

        # partial_fit — pass classes kwarg if supported
        try:
            self.pipeline.partial_fit(X, y, classes=self.classes_)
        except TypeError:
            self.pipeline.partial_fit(X, y)

        elapsed = time.perf_counter() - t0

        # memory estimate: bytes used by this chunk's arrays
        mem_bytes = X.nbytes + y.nbytes

        # training accuracy on this chunk
        train_acc = self._safe_score(X, y)

        self.n_chunks_seen_  += 1
        self.n_samples_seen_ += len(y)

        entry = {
            "chunk":          self.n_chunks_seen_,
            "n_samples":      len(y),
            "n_total":        self.n_samples_seen_,
            "train_acc":      train_acc,
            "eval_acc":       None,
            "cumulative_acc": None,
            "memory_bytes":   mem_bytes,
            "elapsed_s":      round(elapsed, 4),
        }
        self.log_.append(entry)

        if self.verbose:
            self._print_row(entry)

        return self

    def score_chunk(self, X_eval, y_eval):
        """
        Evaluate the current model on a held-out set.

        Updates the most recent log entry with eval_acc and
        cumulative_acc, then returns the current eval accuracy.

        Parameters
        ----------
        X_eval : array-like, shape (n_samples, n_features)
        y_eval : array-like, shape (n_samples,)

        Returns
        -------
        float
            Accuracy on (X_eval, y_eval).

        Raises
        ------
        RuntimeError : if called before any fit_chunk call.
        """
        if not self.log_:
            raise RuntimeError(
                "score_chunk() called before fit_chunk(). "
                "Train on at least one chunk first."
            )

        X_eval = np.asarray(X_eval, dtype=float)
        y_eval = np.asarray(y_eval)

        acc    = self._safe_score(X_eval, y_eval)

        # update cumulative accuracy
        n_correct          = int(np.sum(
            self.pipeline.predict(X_eval) == y_eval
        ))
        self._cum_correct += n_correct
        self._cum_total   += len(y_eval)
        cum_acc            = (
            self._cum_correct / self._cum_total
            if self._cum_total > 0 else 0.0
        )

        # patch most recent log entry
        self.log_[-1]["eval_acc"]       = round(acc,     4)
        self.log_[-1]["cumulative_acc"] = round(cum_acc, 4)

        if self.verbose:
            print(
                f"  → eval_acc={acc:.4f}  "
                f"cumulative_acc={cum_acc:.4f}"
            )

        return acc

    # ── logging helpers ───────────────────────────────────────────────────────

    def summary(self):
        """
        Print a formatted table of the per-chunk training log.

        Columns: chunk, n_samples, n_total, train_acc, eval_acc,
                 cumulative_acc, memory_kb, elapsed_ms
        """
        if not self.log_:
            print("No chunks trained yet.")
            return

        header = (
            f"{'Chunk':>6}  {'N':>6}  {'Total':>7}  "
            f"{'TrainAcc':>9}  {'EvalAcc':>8}  "
            f"{'CumAcc':>7}  {'Mem(KB)':>8}  {'ms':>7}"
        )
        sep = "─" * len(header)
        print(sep)
        print(header)
        print(sep)

        for e in self.log_:
            eval_s = f"{e['eval_acc']:.4f}" if e["eval_acc"] is not None else "    —   "
            cum_s  = f"{e['cumulative_acc']:.4f}" if e["cumulative_acc"] is not None else "   —   "
            print(
                f"{e['chunk']:>6}  "
                f"{e['n_samples']:>6}  "
                f"{e['n_total']:>7}  "
                f"{e['train_acc']:>9.4f}  "
                f"{eval_s:>8}  "
                f"{cum_s:>7}  "
                f"{e['memory_bytes'] / 1024:>8.1f}  "
                f"{e['elapsed_s'] * 1000:>7.1f}"
            )
        print(sep)
        print(
            f"Total chunks: {self.n_chunks_seen_}  |  "
            f"Total samples: {self.n_samples_seen_}"
        )

    def log_to_csv(self, filepath):
        """
        Persist the training log to a CSV file.

        Parameters
        ----------
        filepath : str or Path

        Raises
        ------
        RuntimeError : if no chunks have been trained yet.
        """
        if not self.log_:
            raise RuntimeError("No log entries to write.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "chunk", "n_samples", "n_total",
            "train_acc", "eval_acc", "cumulative_acc",
            "memory_bytes", "elapsed_s",
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(",".join(columns) + "\n")
            for entry in self.log_:
                row = [
                    str(entry.get(col, ""))
                    for col in columns
                ]
                f.write(",".join(row) + "\n")

        print(f"Log saved to {filepath}")

    def get_metric_history(self, metric="eval_acc"):
        """
        Return the history of a logged metric across all chunks.

        Parameters
        ----------
        metric : str, default='eval_acc'
            Any key present in log_ entries.

        Returns
        -------
        list
            Values for the requested metric, skipping None entries.
        """
        return [
            entry[metric]
            for entry in self.log_
            if entry.get(metric) is not None
        ]

    def reset(self):
        """
        Clear all log state and reset counters.

        Does NOT reset the underlying pipeline — call pipeline.fit()
        separately if you want to retrain from scratch.

        Returns
        -------
        self
        """
        self.log_            = []
        self.n_chunks_seen_  = 0
        self.n_samples_seen_ = 0
        self._cum_correct    = 0
        self._cum_total      = 0
        return self

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def latest_eval_acc(self):
        """Most recent eval accuracy, or None if score_chunk not called."""
        for entry in reversed(self.log_):
            if entry["eval_acc"] is not None:
                return entry["eval_acc"]
        return None

    @property
    def latest_train_acc(self):
        """Most recent training chunk accuracy."""
        if not self.log_:
            return None
        return self.log_[-1]["train_acc"]

    @property
    def cumulative_acc(self):
        """Running cumulative accuracy across all score_chunk calls."""
        if self._cum_total == 0:
            return 0.0
        return self._cum_correct / self._cum_total

    # ── internal helpers ──────────────────────────────────────────────────────

    def _safe_score(self, X, y):
        """
        Score the pipeline on (X, y), returning 0.0 on any error.

        Catches RuntimeError so score_chunk is safe to call even
        if the pipeline has only seen one chunk so far.
        """
        try:
            preds = self.pipeline.predict(X)
            return float(np.mean(preds == y))
        except Exception:
            return 0.0

    def _print_row(self, entry):
        """Print a compact one-line progress update."""
        print(
            f"[Chunk {entry['chunk']:>3}]  "
            f"n={entry['n_samples']:>5}  "
            f"total={entry['n_total']:>6}  "
            f"train_acc={entry['train_acc']:.4f}  "
            f"mem={entry['memory_bytes'] / 1024:>7.1f}KB  "
            f"time={entry['elapsed_s'] * 1000:.1f}ms"
        )

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"StreamTrainer("
            f"chunks_seen={self.n_chunks_seen_}, "
            f"samples_seen={self.n_samples_seen_}, "
            f"pipeline={type(self.pipeline).__name__}"
            f")"
        )