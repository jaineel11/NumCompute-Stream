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
from .preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Imputer,
    OneHotEncoder,
)
from .stats import (
    StreamingStats,
    StreamingHistogram,
    ExponentialMovingAverage,
    update_stats,
    reset_stats,
)

__version__ = "0.1.0"