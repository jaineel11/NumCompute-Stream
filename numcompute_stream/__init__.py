"""
numcompute_stream
=================
Streaming, decision tree–based ML framework. NumPy + matplotlib only.
"""

from .io import (
    read_csv,
    write_csv,
    stream_csv,
    split_into_chunks,
    train_test_split,
    make_classification_dataset,
)

__version__ = "0.1.0"