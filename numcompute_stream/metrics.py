"""
metrics.py – Streaming-compatible evaluation metrics.

All metric classes expose a consistent API:

    update(y_true, y_pred)  → ingest a new chunk, update running state
    result()                → return current metric value (spec alias)
    compute()               → identical to result()
    reset()                 → clear all accumulated state

Rolling-window variants maintain a fixed-size deque so metrics reflect
only the most recent N samples rather than all data seen so far.

Classes
-------
Accuracy              – cumulative and rolling accuracy
PrecisionRecallF1     – macro/micro/binary precision, recall, F1
ConfusionMatrix       – incremental confusion matrix, expands on new classes
AUC                   – streaming ROC-AUC via Mann-Whitney U approximation

Module-level batch helpers
--------------------------
accuracy(y_true, y_pred)
precision_recall_f1(y_true, y_pred, average='macro')
confusion_matrix(y_true, y_pred, classes=None)
roc_auc(y_true, y_score)
"""

import numpy as np
from collections import deque


# Accuracy

class Accuracy:
    """
    Compute classification accuracy incrementally.

    Supports both cumulative mode (all data seen so far) and a
    rolling window mode (most recent ``window`` predictions only).

    Parameters
    ----------
    window : int or None, default=None
        If set, only the most recent ``window`` predictions contribute.

    Attributes
    ----------
    n_correct_ : int
    n_total_   : int
    history_   : list[float]
        Accuracy after each update() call.
    """

    def __init__(self, window=None):
        self.window     = window
        self.n_correct_ = 0
        self.n_total_   = 0
        self.history_   = []
        self._buffer    = deque(maxlen=window) if window else None

    def update(self, y_true, y_pred):
        """
        Ingest a new chunk of predictions.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true and y_pred length mismatch: "
                f"{len(y_true)} vs {len(y_pred)}"
            )

        correct = (y_true == y_pred).astype(int)

        if self.window is not None:
            self._buffer.extend(correct.tolist())
            self.n_correct_ = int(np.sum(self._buffer))
            self.n_total_   = len(self._buffer)
        else:
            self.n_correct_ += int(correct.sum())
            self.n_total_   += len(y_true)

        self.history_.append(self.compute())
        return self

    def compute(self):
        """Return current accuracy (0.0 if no data seen)."""
        if self.n_total_ == 0:
            return 0.0
        return self.n_correct_ / self.n_total_

    def result(self):
        """Alias for compute() — matches spec's result() requirement."""
        return self.compute()

    def reset(self):
        """Clear all accumulated state."""
        self.n_correct_ = 0
        self.n_total_   = 0
        self.history_   = []
        if self.window:
            self._buffer = deque(maxlen=self.window)
        return self


# PrecisionRecallF1

class PrecisionRecallF1:
    """
    Incrementally compute precision, recall, and F1.

    Supports macro, micro, and binary averaging, with an optional
    rolling window that only considers the most recent N predictions.

    Parameters
    ----------
    average : str, default='macro'
        'macro'  → unweighted mean over all classes.
        'micro'  → global TP / (TP + FP) etc.
        'binary' → positive class assumed to be 1.
    window : int or None, default=None
        Rolling window size. None → cumulative.

    Attributes
    ----------
    history_ : list[dict]
        Dict with keys precision, recall, f1 after each update().
    """

    def __init__(self, average="macro", window=None):
        if average not in ("macro", "micro", "binary"):
            raise ValueError(
                f"average must be 'macro', 'micro', or 'binary', got '{average}'"
            )
        self.average  = average
        self.window   = window
        self.history_ = []
        self._reset_counts()
        # rolling window buffers store (y_true, y_pred) pairs
        self._buf_true = deque(maxlen=window) if window else None
        self._buf_pred = deque(maxlen=window) if window else None

    def _reset_counts(self):
        self.tp_counts_ = {}
        self.fp_counts_ = {}
        self.fn_counts_ = {}
        self._classes   = set()

    def update(self, y_true, y_pred):
        """
        Ingest a new chunk.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if self.window is not None:
            self._buf_true.extend(y_true.tolist())
            self._buf_pred.extend(y_pred.tolist())
            y_true = np.asarray(self._buf_true)
            y_pred = np.asarray(self._buf_pred)
            # recompute from scratch over window
            self._reset_counts()

        classes = np.union1d(y_true, y_pred)
        self._classes.update(classes.tolist())

        for cls in classes:
            tp = int(np.sum((y_pred == cls) & (y_true == cls)))
            fp = int(np.sum((y_pred == cls) & (y_true != cls)))
            fn = int(np.sum((y_pred != cls) & (y_true == cls)))
            if self.window is None:
                self.tp_counts_[cls] = self.tp_counts_.get(cls, 0) + tp
                self.fp_counts_[cls] = self.fp_counts_.get(cls, 0) + fp
                self.fn_counts_[cls] = self.fn_counts_.get(cls, 0) + fn
            else:
                self.tp_counts_[cls] = tp
                self.fp_counts_[cls] = fp
                self.fn_counts_[cls] = fn

        self.history_.append(self.compute())
        return self

    def compute(self):
        """
        Return current precision, recall, and F1.

        Returns
        -------
        dict with keys: precision, recall, f1
        """
        if not self._classes:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if self.average == "micro":
            total_tp = sum(self.tp_counts_.values())
            total_fp = sum(self.fp_counts_.values())
            total_fn = sum(self.fn_counts_.values())
            p  = total_tp / max(total_tp + total_fp, 1)
            r  = total_tp / max(total_tp + total_fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-10)
            return {"precision": p, "recall": r, "f1": f1}

        target = [1] if self.average == "binary" else sorted(self._classes)
        ps, rs, fs = [], [], []

        for cls in target:
            tp = self.tp_counts_.get(cls, 0)
            fp = self.fp_counts_.get(cls, 0)
            fn = self.fn_counts_.get(cls, 0)
            p  = tp / max(tp + fp, 1)
            r  = tp / max(tp + fn, 1)
            f  = 2 * p * r / max(p + r, 1e-10)
            ps.append(p); rs.append(r); fs.append(f)

        return {
            "precision": float(np.mean(ps)),
            "recall":    float(np.mean(rs)),
            "f1":        float(np.mean(fs)),
        }

    def result(self):
        """Alias for compute()."""
        return self.compute()

    def reset(self):
        """Clear all accumulated state."""
        self._reset_counts()
        self.history_ = []
        if self.window:
            self._buf_true = deque(maxlen=self.window)
            self._buf_pred = deque(maxlen=self.window)
        return self


# ConfusionMatrix

class ConfusionMatrix:
    """
    Incrementally maintain a confusion matrix.

    Automatically expands when new classes are encountered in later
    chunks — no need to specify all classes up front.

    Parameters
    ----------
    classes : array-like or None, default=None
        Known classes. If None, inferred from data.

    Attributes
    ----------
    matrix_  : np.ndarray, shape (n_classes, n_classes)
        Rows = true labels, columns = predicted labels.
    classes_ : np.ndarray
    """

    def __init__(self, classes=None):
        self._classes_hint = np.asarray(classes) if classes is not None else None
        self.classes_      = self._classes_hint
        self.matrix_       = None

    def update(self, y_true, y_pred):
        """
        Add a new chunk to the confusion matrix.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)

        Returns
        -------
        self
        """
        y_true      = np.asarray(y_true)
        y_pred      = np.asarray(y_pred)
        new_classes = np.union1d(y_true, y_pred)

        if self.classes_ is None:
            self.classes_ = new_classes
        else:
            all_classes = np.union1d(self.classes_, new_classes)
            if len(all_classes) > len(self.classes_):
                # expand existing matrix
                n_new    = len(all_classes)
                expanded = np.zeros((n_new, n_new), dtype=int)
                old_idx  = [
                    int(np.searchsorted(all_classes, c))
                    for c in self.classes_
                ]
                if self.matrix_ is not None:
                    for oi, ni in enumerate(old_idx):
                        for oj, nj in enumerate(old_idx):
                            expanded[ni, nj] = self.matrix_[oi, oj]
                self.matrix_  = expanded
                self.classes_ = all_classes

        n = len(self.classes_)
        if self.matrix_ is None:
            self.matrix_ = np.zeros((n, n), dtype=int)

        true_idx = np.searchsorted(self.classes_, y_true)
        pred_idx = np.searchsorted(self.classes_, y_pred)
        np.add.at(self.matrix_, (true_idx, pred_idx), 1)
        return self

    def compute(self):
        """Return a copy of the current confusion matrix."""
        if self.matrix_ is None:
            return np.array([[]])
        return self.matrix_.copy()

    def result(self):
        """Alias for compute()."""
        return self.compute()

    def reset(self):
        """Clear accumulated state."""
        self.classes_ = self._classes_hint
        self.matrix_  = None
        return self


# AUC  (ROC-AUC via Mann-Whitney U)

class AUC:
    """
    Streaming ROC-AUC for binary classification.

    Accumulates (y_true, y_score) pairs across chunks and recomputes
    AUC from all buffered data on each update. Uses the Mann-Whitney U
    statistic which is equivalent to the trapezoidal ROC-AUC.

    Parameters
    ----------
    window : int or None, default=None
        If set, only the most recent ``window`` samples contribute.

    Attributes
    ----------
    history_ : list[float]
        AUC after each update() call.
    """

    def __init__(self, window=None):
        self.window   = window
        self.history_ = []
        self._buf_true  = deque(maxlen=window) if window else []
        self._buf_score = deque(maxlen=window) if window else []

    def update(self, y_true, y_score):
        """
        Add a new chunk of true labels and predicted scores.

        Parameters
        ----------
        y_true  : array-like, shape (n,)
            Binary labels. Exactly two unique values required.
        y_score : array-like, shape (n,)
            Predicted probability or decision score for the positive class.

        Returns
        -------
        self
        """
        y_true  = np.asarray(y_true)
        y_score = np.asarray(y_score, dtype=float)

        if isinstance(self._buf_true, list):
            self._buf_true.extend(y_true.tolist())
            self._buf_score.extend(y_score.tolist())
        else:
            self._buf_true.extend(y_true.tolist())
            self._buf_score.extend(y_score.tolist())

        self.history_.append(self.compute())
        return self

    def compute(self):
        """
        Return current ROC-AUC score.

        Returns
        -------
        float in [0, 1], or 0.0 if fewer than 2 classes seen.
        """
        if len(self._buf_true) == 0:
            return 0.0

        yt = np.asarray(self._buf_true)
        ys = np.asarray(self._buf_score)

        classes = np.unique(yt)
        if len(classes) < 2:
            return 0.0

        # detect positive class dynamically (larger label = positive)
        neg_cls, pos_cls = classes[0], classes[1]
        pos_scores = ys[yt == pos_cls]
        neg_scores = ys[yt == neg_cls]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0

        # Mann-Whitney U: fraction of (pos, neg) pairs where pos > neg
        # vectorised via broadcasting
        u = np.mean(
            pos_scores[:, None] > neg_scores[None, :]
        ) + 0.5 * np.mean(
            pos_scores[:, None] == neg_scores[None, :]
        )
        return float(u)

    def result(self):
        """Alias for compute()."""
        return self.compute()

    def reset(self):
        """Clear accumulated state."""
        self.history_   = []
        self._buf_true  = deque(maxlen=self.window) if self.window else []
        self._buf_score = deque(maxlen=self.window) if self.window else []
        return self


# Batch convenience functions

def accuracy(y_true, y_pred):
    """
    Batch accuracy.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("y_true is empty.")
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, average="macro"):
    """
    Batch precision, recall, and F1.

    Parameters
    ----------
    y_true   : array-like
    y_pred   : array-like
    average  : str, default='macro'

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    m = PrecisionRecallF1(average=average)
    m.update(y_true, y_pred)
    return m.compute()


def confusion_matrix(y_true, y_pred, classes=None):
    """
    Batch confusion matrix.

    Parameters
    ----------
    y_true  : array-like
    y_pred  : array-like
    classes : array-like or None

    Returns
    -------
    matrix  : np.ndarray
    classes : np.ndarray
    """
    cm = ConfusionMatrix(classes=classes)
    cm.update(y_true, y_pred)
    return cm.compute(), cm.classes_


def roc_auc(y_true, y_score):
    """
    Batch ROC-AUC score.

    Parameters
    ----------
    y_true  : array-like, binary labels
    y_score : array-like, predicted scores

    Returns
    -------
    float
    """
    a = AUC()
    a.update(y_true, y_score)
    return a.compute()