# numcompute_stream

A streaming decision tree based machine learning framework built on NumPy only
(plus matplotlib for visualisation). Extends the NumCompute package to support
online and incremental learning via partial_fit on every component.

## Features

- **DecisionTreeClassifier** with Gini and entropy support, depth limiting and partial_fit
- **EnsembleClassifier** supporting bagging, random_forest and adaboost via one unified class
- **StandardScaler and MinMaxScaler** using Welford online algorithm for streaming updates
- **Imputer** with mean, median and constant strategies all updated incrementally
- **OneHotEncoder** that expands categories as new ones appear in the stream
- **Accuracy, PrecisionRecallF1, ConfusionMatrix and AUC** all with rolling window support
- **StreamingStats and StreamingHistogram** with optional sliding window
- **Pipeline** that chains transformers and estimator with partial_fit throughout
- **StreamTrainer** that manages the training loop with per chunk logging and summary table
- **io** for CSV reading, streaming CSV generator, chunk splitting and dataset generation
- **visualise** with plot_metric_over_time, compare_models and plot_predictions_vs_ground_truth

## Quick Start

```python
from numcompute_stream import (
    Pipeline, StandardScaler, EnsembleClassifier, StreamTrainer
)
from numcompute_stream.io import read_csv, split_into_chunks, train_test_split

headers, data = read_csv("demo/data/dataset.csv")
X, y = data[:, :-1], data[:, -1].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", EnsembleClassifier(method="random_forest", n_estimators=10, max_depth=5)),
])

trainer = StreamTrainer(pipeline=pipe, verbose=True)

for Xc, yc in split_into_chunks(X_tr, y_tr, n_chunks=10):
    trainer.fit_chunk(Xc, yc)
    print(trainer.score_chunk(X_te, y_te))

trainer.summary()
```

## Setup

```bash
pip install numpy matplotlib pytest
```

## Tests

```bash
pytest tests/ -v
```

70 unit tests across 18 test classes covering all modules. Includes streaming
specific edge cases like NaN inputs, zero variance columns, single class datasets,
consistency between fit and sequential partial_fit calls, and the result() alias
on every metric class.

## Demo

```bash
python demo/stream_demo.py
```

Loads a CSV dataset, streams through 10 chunks, trains three pipelines
(DecisionTree, RandomForest, AdaBoost), logs metrics via StreamTrainer and saves
visualisation plots to demo/outputs/.

## Benchmarks

```bash
python benchmark/run_benchmarks.py
```

Results from a real run on Windows, Python 3.12, NumPy 2.x:

**Single tree vs ensemble (streaming, 5 chunks)**

| Model | Time (ms) | Test Accuracy |
|-------|-----------|---------------|
| DecisionTreeClassifier (depth=5, gini) | 969.2 | 1.000 |
| EnsembleClassifier method=bagging n=5 | 3006.3 | 1.000 |
| EnsembleClassifier method=random_forest n=5 | 895.4 | 1.000 |
| EnsembleClassifier method=adaboost n=5 | 3033.5 | 1.000 |

Random Forest was actually faster than the single tree because it only considers
sqrt(d) features per split so the cost per tree is much lower. Bagging and AdaBoost
were around 3x slower.

**Chunk size sensitivity (RandomForest, 3 estimators)**

| Chunk Size | N Chunks | Test Accuracy | Time (ms) |
|------------|----------|---------------|-----------|
| 20 | 24 | 1.000 | 3065.2 |
| 50 | 9 | 1.000 | 1246.8 |
| 100 | 4 | 1.000 | 771.2 |
| 200 | 2 | 1.000 | 471.1 |

**Gini vs Entropy (DecisionTreeClassifier, depth=5)**

| Criterion | Time (ms) | Test Accuracy |
|-----------|-----------|---------------|
| gini | 1861.6 | 1.000 |
| entropy | 2275.3 | 1.000 |

Entropy is about 22 percent slower due to log2 computation at every split.
Both criteria gave the same accuracy on this dataset.

## Module Overview

| Module | Key Classes and Functions |
|--------|--------------------------|
| `tree.py` | `DecisionTreeClassifier` |
| `ensemble.py` | `EnsembleClassifier` with method=bagging, random_forest, adaboost |
| `preprocessing.py` | `StandardScaler`, `MinMaxScaler`, `Imputer`, `OneHotEncoder` |
| `metrics.py` | `Accuracy`, `PrecisionRecallF1`, `ConfusionMatrix`, `AUC` |
| `stats.py` | `StreamingStats`, `StreamingHistogram`, `ExponentialMovingAverage` |
| `pipeline.py` | `Pipeline` |
| `stream.py` | `StreamTrainer` |
| `io.py` | `read_csv`, `write_csv`, `stream_csv`, `split_into_chunks`, `train_test_split`, `make_classification_dataset` |
| `visualise.py` | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth`, `plot_confusion_matrix`, `plot_learning_curve` |

## Design Decisions

- **Rebuild on partial_fit**: Trees buffer all data and rebuild from scratch on each
  partial_fit call to guarantee exact optimal splits. O(N) cost per chunk which is
  fine for typical streaming batch sizes.
- **Unified EnsembleClassifier**: All three ensemble methods share one class with a
  method parameter since they all follow the same pattern of training multiple trees
  and combining votes.
- **Welford algorithm**: All scalers and StreamingStats use Welford online algorithm
  for numerically stable mean and variance without catastrophic cancellation.
- **NaN safety**: Every entry point handles NaN before any computation. Trees use
  column median imputation, scalers substitute the running mean, histograms filter
  before binning.
- **Pure NumPy**: No scikit-learn, pandas or any other ML library. Only numpy
  and matplotlib.

## Requirements

- Python 3.10 or higher
- numpy 1.26 or higher
- matplotlib 3.7 or higher