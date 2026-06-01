"""
io.py – Streaming-compatible I/O utilities.

Provides chunk-wise CSV reading and dataset splitting for simulating
streaming data scenarios. No pandas — pure Python + NumPy only.
"""

import numpy as np
import csv
from pathlib import Path


def read_csv(filepath, delimiter=",", has_header=True, dtype=float, missing_val=np.nan):
    """
    Read a CSV file into a NumPy array.

    Parameters
    ----------
    filepath : str or Path
    delimiter : str, default=','
    has_header : bool, default=True
    dtype : type, default=float
    missing_val : scalar, default=np.nan
        Value to substitute for unparseable fields.

    Returns
    -------
    headers : list[str]
        Column names (empty list if has_header=False).
    data : np.ndarray, shape (n_rows, n_cols)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")

    headers = []
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            headers = next(reader)
        for row in reader:
            parsed = []
            for val in row:
                val = val.strip()
                try:
                    parsed.append(dtype(val))
                except (ValueError, TypeError):
                    parsed.append(missing_val)
            rows.append(parsed)

    if not rows:
        return headers, np.empty((0, len(headers)), dtype=float)

    return headers, np.array(rows, dtype=float)


def write_csv(filepath, data, headers=None, delimiter=","):
    """
    Write a NumPy array to a CSV file.

    Parameters
    ----------
    filepath : str or Path
    data : np.ndarray, shape (n_rows, n_cols)
    headers : list[str] or None
    delimiter : str
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        if headers:
            writer.writerow(headers)
        for row in data:
            writer.writerow([format(float(v), "g") for v in row])


def stream_csv(filepath, chunk_size=100, delimiter=",", has_header=True,
               dtype=float, missing_val=np.nan):
    """
    Generator that yields chunks of rows from a CSV file.

    Simulates a real streaming data source — never loads the full
    file into memory at once.

    Parameters
    ----------
    filepath : str or Path
    chunk_size : int, default=100
    delimiter : str
    has_header : bool
    dtype : type
    missing_val : scalar

    Yields
    ------
    headers : list[str]
    chunk : np.ndarray, shape (<=chunk_size, n_cols)

    Example
    -------
    >>> for headers, chunk in stream_csv("data.csv", chunk_size=50):
    ...     print(chunk.shape)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")

    headers = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            headers = next(reader)

        chunk = []
        for row in reader:
            parsed = []
            for val in row:
                val = val.strip()
                try:
                    parsed.append(dtype(val))
                except (ValueError, TypeError):
                    parsed.append(missing_val)
            chunk.append(parsed)
            if len(chunk) == chunk_size:
                yield headers, np.array(chunk, dtype=float)
                chunk = []

        if chunk:
            yield headers, np.array(chunk, dtype=float)


def split_into_chunks(X, y, n_chunks=10, shuffle=True, random_state=None):
    """
    Split arrays into equal-sized chunks to simulate a streaming scenario.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    n_chunks : int, default=10
    shuffle : bool, default=True
    random_state : int or None

    Returns
    -------
    list of (X_chunk, y_chunk) tuples
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    if shuffle:
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(y))
        X, y = X[idx], y[idx]

    chunk_indices = np.array_split(np.arange(len(y)), n_chunks)
    return [(X[idx], y[idx]) for idx in chunk_indices]


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split data into train and test sets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    test_size : float, default=0.2
        Fraction of samples for the test set.
    shuffle : bool, default=True
    random_state : int or None

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(y)

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    if shuffle:
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(n)
        X, y = X[idx], y[idx]

    n_test = max(1, int(n * test_size))
    return X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]


def make_classification_dataset(
    n_samples=500,
    n_features=10,
    n_classes=2,
    n_informative=5,
    noise=0.1,
    random_state=None,
):
    """
    Generate a synthetic classification dataset (no sklearn required).

    Informative features are drawn from class-specific Gaussian centroids.
    Remaining features are pure noise.

    Parameters
    ----------
    n_samples : int
    n_features : int
    n_classes : int
    n_informative : int
    noise : float
        Standard deviation of Gaussian noise.
    random_state : int or None

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    n_informative = min(n_informative, n_features)

    centroids = rng.randn(n_classes, n_informative) * 2.0
    X = np.zeros((n_samples, n_features))
    y = rng.randint(0, n_classes, size=n_samples)

    for cls in range(n_classes):
        mask = y == cls
        X[mask, :n_informative] = (
            centroids[cls] + rng.randn(mask.sum(), n_informative) * noise
        )

    if n_features > n_informative:
        X[:, n_informative:] = rng.randn(n_samples, n_features - n_informative) * noise

    return X, y