"""
tests/test_all.py – Full test suite for numcompute_stream.

70+ unit tests across all modules. Run with:
    pytest tests/ -v

Coverage
--------
    TestDecisionTreeClassifier  – 10 tests
    TestEnsembleClassifier      – 12 tests
    TestStandardScaler          –  5 tests
    TestMinMaxScaler            –  3 tests
    TestImputer                 –  5 tests
    TestOneHotEncoder           –  4 tests
    TestStreamingStats          –  5 tests
    TestStreamingHistogram      –  4 tests
    TestEWMA                    –  4 tests
    TestAccuracy                –  5 tests
    TestPrecisionRecallF1       –  3 tests
    TestConfusionMatrix         –  3 tests
    TestAUC                     –  3 tests
    TestBatchMetrics            –  3 tests
    TestPipeline                –  5 tests
    TestStreamTrainer           –  6 tests
    TestIO                      –  5 tests
    TestEdgeCases               –  6 tests
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.preprocessing import (
    StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
)
from numcompute_stream.stats import (
    StreamingStats, StreamingHistogram, ExponentialMovingAverage,
    update_stats, reset_stats,
)
from numcompute_stream.metrics import (
    Accuracy, PrecisionRecallF1, ConfusionMatrix, AUC,
    accuracy, precision_recall_f1, confusion_matrix, roc_auc,
)
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.io import (
    split_into_chunks, train_test_split, make_classification_dataset,
    read_csv, write_csv,
)


# Fixtures

@pytest.fixture
def binary_data():
    rng = np.random.RandomState(42)
    X   = rng.randn(200, 4)
    y   = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    rng = np.random.RandomState(0)
    X   = rng.randn(300, 4)
    y   = rng.randint(0, 3, size=300)
    return X, y


@pytest.fixture
def simple_pipe(binary_data):
    X, y = binary_data
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  DecisionTreeClassifier(max_depth=3, random_state=0)),
    ])
    pipe.fit(X, y)
    return pipe, X, y


# ═════════════════════════════════════════════════════════════════════════════
# DecisionTreeClassifier
# ═════════════════════════════════════════════════════════════════════════════

class TestDecisionTreeClassifier:

    def test_fit_predict_binary(self, binary_data):
        X, y = binary_data
        tree = DecisionTreeClassifier(max_depth=3, random_state=1)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert preds.shape == (200,)
        assert set(preds).issubset({0, 1})
        assert tree.score(X, y) > 0.6

    def test_partial_fit_accumulates(self, binary_data):
        X, y   = binary_data
        chunks = split_into_chunks(X, y, n_chunks=5, random_state=0)
        tree   = DecisionTreeClassifier(max_depth=4, random_state=1)
        for Xc, yc in chunks:
            tree.partial_fit(Xc, yc)
        assert tree.n_samples_seen_ == 200
        assert tree.score(X, y) > 0.55

    def test_entropy_criterion(self, binary_data):
        X, y = binary_data
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        tree.fit(X, y)
        assert tree.score(X, y) > 0.6

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError, match="criterion"):
            DecisionTreeClassifier(criterion="bad")

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y  = binary_data
        tree  = DecisionTreeClassifier(max_depth=3, random_state=1)
        tree.fit(X, y)
        proba = tree.predict_proba(X)
        assert proba.shape == (200, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_nan_input_handled(self):
        rng   = np.random.RandomState(7)
        X     = rng.randn(100, 3)
        y     = (X[:, 0] > 0).astype(int)
        X[::5, 0] = np.nan
        tree  = DecisionTreeClassifier(max_depth=3, random_state=0)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert preds.shape == (100,)

    def test_depth_respected(self, binary_data):
        X, y = binary_data
        tree = DecisionTreeClassifier(max_depth=2, random_state=1)
        tree.fit(X, y)
        assert tree.get_depth() <= 2

    def test_unfitted_raises(self, binary_data):
        X, _ = binary_data
        tree = DecisionTreeClassifier()
        with pytest.raises(RuntimeError, match="fitted"):
            tree.predict(X)

    def test_x_must_be_2d(self):
        tree = DecisionTreeClassifier()
        with pytest.raises(ValueError, match="2-D"):
            tree.fit(np.array([1, 2, 3]), np.array([0, 1, 0]))

    def test_multiclass_predict(self, multiclass_data):
        X, y = multiclass_data
        tree = DecisionTreeClassifier(max_depth=5, random_state=0)
        tree.fit(X, y)
        assert set(tree.predict(X)).issubset({0, 1, 2})


# ═════════════════════════════════════════════════════════════════════════════
# EnsembleClassifier
# ═════════════════════════════════════════════════════════════════════════════

class TestEnsembleClassifier:

    def test_bagging_fit_predict(self, binary_data):
        X, y = binary_data
        ens  = EnsembleClassifier(method="bagging", n_estimators=5,
                                   random_state=0)
        ens.fit(X, y)
        assert ens.score(X, y) > 0.6

    def test_random_forest_fit_predict(self, binary_data):
        X, y = binary_data
        ens  = EnsembleClassifier(method="random_forest", n_estimators=5,
                                   random_state=0)
        ens.fit(X, y)
        assert ens.score(X, y) > 0.6

    def test_adaboost_fit_predict(self, binary_data):
        X, y = binary_data
        ens  = EnsembleClassifier(method="adaboost", n_estimators=5,
                                   random_state=0)
        ens.fit(X, y)
        assert ens.score(X, y) > 0.55

    def test_partial_fit_streaming(self, binary_data):
        X, y   = binary_data
        chunks = split_into_chunks(X, y, n_chunks=4, random_state=0)
        ens    = EnsembleClassifier(method="random_forest", n_estimators=5,
                                    random_state=0)
        for Xc, yc in chunks:
            ens.partial_fit(Xc, yc)
        assert ens.n_samples_seen_ == 200

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y  = binary_data
        ens   = EnsembleClassifier(method="bagging", n_estimators=3,
                                    random_state=0)
        ens.fit(X, y)
        proba = ens.predict_proba(X)
        assert proba.shape[0] == 200
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_estimator_count(self, binary_data):
        X, y = binary_data
        ens  = EnsembleClassifier(n_estimators=7, random_state=0)
        ens.fit(X, y)
        assert len(ens.estimators_) == 7

    def test_adaboost_weights_positive(self, binary_data):
        X, y = binary_data
        ens  = EnsembleClassifier(method="adaboost", n_estimators=5,
                                   random_state=0)
        ens.fit(X, y)
        assert np.all(ens.estimator_weights_ > 0)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            EnsembleClassifier(method="not_a_method")

    def test_unfitted_raises(self, binary_data):
        X, _ = binary_data
        ens  = EnsembleClassifier()
        with pytest.raises(RuntimeError, match="fitted"):
            ens.predict(X)

    def test_multiclass_random_forest(self, multiclass_data):
        X, y = multiclass_data
        ens  = EnsembleClassifier(method="random_forest", n_estimators=5,
                                   random_state=0)
        ens.fit(X, y)
        assert set(ens.predict(X)).issubset({0, 1, 2})

    def test_entropy_criterion(self, binary_data):
        X, y = binary_data
        ens  = EnsembleClassifier(
            method="bagging", n_estimators=3,
            criterion="entropy", random_state=0,
        )
        ens.fit(X, y)
        assert ens.score(X, y) > 0.5

    def test_partial_fit_multiclass(self, multiclass_data):
        X, y   = multiclass_data
        chunks = split_into_chunks(X, y, n_chunks=5, random_state=0)
        ens    = EnsembleClassifier(method="random_forest", n_estimators=3,
                                    random_state=0)
        for Xc, yc in chunks:
            ens.partial_fit(Xc, yc)
        assert ens.score(X, y) > 0.3


# ═════════════════════════════════════════════════════════════════════════════
# StandardScaler
# ═════════════════════════════════════════════════════════════════════════════

class TestStandardScaler:

    def test_zero_mean_after_fit(self):
        rng = np.random.RandomState(0)
        X   = rng.randn(100, 3) * 5 + 3
        s   = StandardScaler()
        Xt  = s.fit_transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), 0, atol=0.05)

    def test_partial_fit_matches_fit(self):
        rng = np.random.RandomState(1)
        X   = rng.randn(100, 3)
        s1  = StandardScaler().fit(X)
        s2  = StandardScaler()
        for chunk in np.array_split(X, 5):
            s2.partial_fit(chunk)
        np.testing.assert_allclose(s1.mean_, s2.mean_, atol=0.01)

    def test_nan_replaced_with_zero(self):
        rng = np.random.RandomState(2)
        X   = rng.randn(50, 2)
        X[::10, 0] = np.nan
        s   = StandardScaler()
        s.fit(X)
        Xt  = s.transform(X)
        assert not np.any(np.isnan(Xt))

    def test_constant_column_no_error(self):
        X        = np.ones((50, 2))
        X[:, 1]  = np.random.randn(50)
        s        = StandardScaler()
        s.fit(X)
        Xt = s.transform(X)
        assert np.all(np.isfinite(Xt))

    def test_unfitted_raises(self):
        s = StandardScaler()
        with pytest.raises(RuntimeError, match="fitted"):
            s.transform(np.ones((5, 2)))


# ═════════════════════════════════════════════════════════════════════════════
# MinMaxScaler
# ═════════════════════════════════════════════════════════════════════════════

class TestMinMaxScaler:

    def test_range_0_1(self):
        rng = np.random.RandomState(0)
        X   = rng.randn(100, 3) * 10
        s   = MinMaxScaler(feature_range=(0, 1))
        Xt  = s.fit_transform(X)
        assert Xt.min() >= -1e-9
        assert Xt.max() <= 1 + 1e-9

    def test_running_min_max(self):
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[0.0, 5.0], [2.0, 3.0]])
        s  = MinMaxScaler()
        s.partial_fit(X1)
        s.partial_fit(X2)
        np.testing.assert_array_equal(s.min_, [0.0, 2.0])
        np.testing.assert_array_equal(s.max_, [3.0, 5.0])

    def test_constant_column_no_error(self):
        X  = np.ones((10, 2))
        s  = MinMaxScaler()
        s.fit(X)
        Xt = s.transform(X)
        assert np.all(np.isfinite(Xt))


# ═════════════════════════════════════════════════════════════════════════════
# Imputer
# ═════════════════════════════════════════════════════════════════════════════

class TestImputer:

    def test_mean_strategy(self):
        X    = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 2.0]])
        imp  = Imputer(strategy="mean")
        Xt   = imp.fit_transform(X)
        assert not np.any(np.isnan(Xt))
        # column 0 mean = 3.0, col 1 mean = 3.0
        assert abs(Xt[0, 1] - 3.0) < 0.01

    def test_constant_strategy(self):
        X   = np.array([[np.nan, 1.0], [2.0, np.nan]])
        imp = Imputer(strategy="constant", fill_value=-1.0)
        Xt  = imp.fit_transform(X)
        assert Xt[0, 0] == -1.0
        assert Xt[1, 1] == -1.0

    def test_median_strategy(self):
        X   = np.array([[1.0, np.nan], [2.0, 4.0], [3.0, 6.0]])
        imp = Imputer(strategy="median")
        Xt  = imp.fit_transform(X)
        assert not np.any(np.isnan(Xt))

    def test_partial_fit_streaming(self):
        rng = np.random.RandomState(0)
        X   = rng.randn(100, 3)
        X[::5, 0] = np.nan
        imp = Imputer(strategy="mean")
        for chunk in np.array_split(X, 5):
            imp.partial_fit(chunk)
        Xt = imp.transform(X)
        assert not np.any(np.isnan(Xt))

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            Imputer(strategy="mode")


# ═════════════════════════════════════════════════════════════════════════════
# OneHotEncoder
# ═════════════════════════════════════════════════════════════════════════════

class TestOneHotEncoder:

    def test_basic_encoding(self):
        X   = np.array([[0], [1], [2], [0]])
        enc = OneHotEncoder()
        Xt  = enc.fit_transform(X)
        assert Xt.shape == (4, 3)
        np.testing.assert_array_equal(Xt.sum(axis=1), 1)

    def test_inverse_transform(self):
        X   = np.array([[0], [1], [2]])
        enc = OneHotEncoder()
        enc.fit(X)
        Xt  = enc.transform(X)
        X2  = enc.inverse_transform(Xt)
        np.testing.assert_array_equal(X2.flatten(), X.flatten())

    def test_partial_fit_new_categories(self):
        enc = OneHotEncoder()
        enc.partial_fit(np.array([[0], [1]]))
        enc.partial_fit(np.array([[2]]))
        assert len(enc.categories_[0]) == 3

    def test_unknown_category_raises(self):
        enc = OneHotEncoder()
        enc.fit(np.array([[0], [1]]))
        with pytest.raises(ValueError, match="unknown"):
            enc.transform(np.array([[5]]))


# ═════════════════════════════════════════════════════════════════════════════
# StreamingStats
# ═════════════════════════════════════════════════════════════════════════════

class TestStreamingStats:

    def test_mean_1d(self):
        ss   = StreamingStats()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ss.update(data)
        assert abs(ss.mean_[0] - 3.0) < 1e-9

    def test_incremental_matches_batch(self):
        rng  = np.random.RandomState(0)
        data = rng.randn(200)
        ss_b = StreamingStats(ddof=1)
        ss_b.update(data)
        ss_i = StreamingStats(ddof=1)
        for chunk in np.array_split(data, 10):
            ss_i.update(chunk)
        assert abs(ss_b.mean_[0] - ss_i.mean_[0]) < 0.01

    def test_nan_does_not_corrupt(self):
        ss   = StreamingStats()
        data = np.array([1.0, np.nan, 3.0])
        ss.update(data)
        assert np.isfinite(ss.mean_[0])

    def test_2d_input(self):
        rng = np.random.RandomState(1)
        X   = rng.randn(100, 3)
        ss  = StreamingStats()
        ss.update(X)
        assert ss.mean_.shape == (3,)
        assert ss.min_.shape  == (3,)

    def test_update_stats_module_function(self):
        reset_stats()
        rng  = np.random.RandomState(0)
        data = rng.randn(50)
        out  = update_stats(data)
        assert "mean" in out
        assert "std"  in out
        assert out["n"] == 50


# ═════════════════════════════════════════════════════════════════════════════
# StreamingHistogram
# ═════════════════════════════════════════════════════════════════════════════

class TestStreamingHistogram:

    def test_counts_correct(self):
        sh = StreamingHistogram(bins=5, range_=(0, 10))
        sh.update(np.array([1.0, 2.0, 5.0, 9.0]))
        counts, _ = sh.compute()
        assert counts.sum() == 4

    def test_incremental_accumulation(self):
        sh = StreamingHistogram(bins=5, range_=(0, 10))
        sh.update(np.array([1.0, 2.0]))
        sh.update(np.array([3.0, 4.0]))
        counts, _ = sh.compute()
        assert counts.sum() == 4

    def test_nan_ignored(self):
        sh = StreamingHistogram(bins=5, range_=(0, 10))
        sh.update(np.array([1.0, np.nan, 3.0]))
        counts, _ = sh.compute()
        assert counts.sum() == 2

    def test_sliding_window(self):
        sh = StreamingHistogram(bins=5, range_=(0, 10), window=4)
        sh.update(np.array([1.0, 2.0, 3.0, 4.0]))
        sh.update(np.array([5.0, 6.0]))   # window drops 1.0, 2.0
        counts, _ = sh.compute()
        assert counts.sum() == 4


# ═════════════════════════════════════════════════════════════════════════════
# ExponentialMovingAverage
# ═════════════════════════════════════════════════════════════════════════════

class TestEWMA:

    def test_initial_value(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        ema.update(10.0)
        assert ema.compute() == 10.0

    def test_smoothing(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        ema.update(10.0)
        ema.update(0.0)
        assert abs(ema.compute() - 5.0) < 1e-9

    def test_nan_ignored(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        ema.update(5.0)
        ema.update(np.nan)
        assert ema.compute() == 5.0

    def test_history_length(self):
        ema = ExponentialMovingAverage(alpha=0.3)
        for v in [1, 2, 3, 4, 5]:
            ema.update(v)
        assert len(ema.history_) == 5


# ═════════════════════════════════════════════════════════════════════════════
# Accuracy
# ═════════════════════════════════════════════════════════════════════════════

class TestAccuracy:

    def test_perfect(self):
        m = Accuracy()
        y = np.array([0, 1, 0, 1])
        m.update(y, y)
        assert m.compute() == 1.0

    def test_incremental(self):
        m = Accuracy()
        m.update([0, 1, 0], [0, 1, 1])   # 2/3
        m.update([1, 1],    [1, 0])       # 1/2 → total 3/5
        assert abs(m.compute() - 0.6) < 1e-9

    def test_result_alias(self):
        m = Accuracy()
        m.update([0, 1], [0, 1])
        assert m.result() == m.compute()

    def test_rolling_window(self):
        m = Accuracy(window=4)
        m.update([1, 1, 1, 1], [1, 1, 1, 0])   # 3/4
        m.update([1, 1],       [1, 1])           # window: last 4 = [1,0,1,1] → 3/4
        assert 0 < m.compute() <= 1.0

    def test_reset(self):
        m = Accuracy()
        m.update([0, 1], [0, 1])
        m.reset()
        assert m.compute() == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# PrecisionRecallF1
# ═════════════════════════════════════════════════════════════════════════════

class TestPrecisionRecallF1:

    def test_perfect(self):
        m = PrecisionRecallF1()
        y = np.array([0, 1, 0, 1])
        m.update(y, y)
        r = m.compute()
        assert r["f1"] == 1.0

    def test_result_alias(self):
        m = PrecisionRecallF1()
        m.update([0, 1], [0, 1])
        assert m.result() == m.compute()

    def test_rolling_window(self):
        m = PrecisionRecallF1(window=4)
        m.update([0, 1, 0, 1], [0, 1, 1, 0])
        r = m.compute()
        assert 0 <= r["f1"] <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# ConfusionMatrix
# ═════════════════════════════════════════════════════════════════════════════

class TestConfusionMatrix:

    def test_shape(self):
        cm = ConfusionMatrix()
        cm.update([0, 1, 0, 1], [0, 1, 1, 0])
        assert cm.compute().shape == (2, 2)

    def test_perfect_diagonal(self):
        cm = ConfusionMatrix()
        y  = np.array([0, 1, 2])
        cm.update(y, y)
        np.testing.assert_array_equal(cm.compute(), np.diag([1, 1, 1]))

    def test_result_alias(self):
        cm = ConfusionMatrix()
        cm.update([0, 1], [0, 1])
        np.testing.assert_array_equal(cm.result(), cm.compute())


# ═════════════════════════════════════════════════════════════════════════════
# AUC
# ═════════════════════════════════════════════════════════════════════════════

class TestAUC:

    def test_perfect_auc(self):
        a = AUC()
        a.update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert abs(a.compute() - 1.0) < 1e-9

    def test_random_auc_near_half(self):
        rng = np.random.RandomState(0)
        a   = AUC()
        y   = rng.randint(0, 2, 200)
        s   = rng.rand(200)
        a.update(y, s)
        assert 0.3 < a.compute() < 0.7

    def test_result_alias(self):
        a = AUC()
        a.update([0, 1], [0.2, 0.8])
        assert a.result() == a.compute()


# ═════════════════════════════════════════════════════════════════════════════
# Batch metric helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestBatchMetrics:

    def test_accuracy_batch(self):
        assert accuracy([0, 1, 0], [0, 1, 0]) == 1.0
        assert accuracy([0, 1, 0], [1, 0, 1]) == 0.0

    def test_f1_batch(self):
        r = precision_recall_f1([0, 1, 0, 1], [0, 1, 0, 1])
        assert r["f1"] == 1.0

    def test_roc_auc_batch(self):
        score = roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert abs(score - 1.0) < 1e-9


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TestPipeline:

    def test_fit_score(self, binary_data):
        X, y = binary_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("tree",   DecisionTreeClassifier(max_depth=3, random_state=0)),
        ])
        pipe.fit(X, y)
        assert pipe.score(X, y) > 0.55

    def test_partial_fit_logs_metrics(self, binary_data):
        X, y   = binary_data
        pipe   = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  EnsembleClassifier(n_estimators=3, random_state=0)),
        ])
        chunks = split_into_chunks(X, y, n_chunks=5, random_state=0)
        for Xc, yc in chunks:
            pipe.partial_fit(Xc, yc, X_eval=X, y_eval=y)
        assert len(pipe.metric_log_) == 5

    def test_predict_proba(self, simple_pipe):
        pipe, X, y = simple_pipe
        proba      = pipe.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Pipeline([])

    def test_repr(self, simple_pipe):
        pipe, _, _ = simple_pipe
        r = repr(pipe)
        assert "Pipeline" in r
        assert "scaler"   in r


# ═════════════════════════════════════════════════════════════════════════════
# StreamTrainer
# ═════════════════════════════════════════════════════════════════════════════

class TestStreamTrainer:

    def _make_trainer(self, binary_data):
        X, y = binary_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeClassifier(max_depth=3, random_state=0)),
        ])
        return StreamTrainer(pipeline=pipe), X, y

    def test_fit_chunk_increments_counter(self, binary_data):
        trainer, X, y = self._make_trainer(binary_data)
        chunks = split_into_chunks(X, y, n_chunks=4, random_state=0)
        for Xc, yc in chunks:
            trainer.fit_chunk(Xc, yc)
        assert trainer.n_chunks_seen_  == 4
        assert trainer.n_samples_seen_ == 200

    def test_score_chunk_returns_float(self, binary_data):
        trainer, X, y = self._make_trainer(binary_data)
        chunks = split_into_chunks(X, y, n_chunks=4, random_state=0)
        trainer.fit_chunk(*chunks[0])
        acc = trainer.score_chunk(X, y)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_log_has_correct_keys(self, binary_data):
        trainer, X, y = self._make_trainer(binary_data)
        trainer.fit_chunk(X[:50], y[:50])
        entry = trainer.log_[0]
        for key in ("chunk", "n_samples", "n_total",
                    "train_acc", "memory_bytes", "elapsed_s"):
            assert key in entry

    def test_score_chunk_before_fit_raises(self, binary_data):
        trainer, X, y = self._make_trainer(binary_data)
        with pytest.raises(RuntimeError, match="fit_chunk"):
            trainer.score_chunk(X, y)

    def test_get_metric_history(self, binary_data):
        trainer, X, y = self._make_trainer(binary_data)
        chunks = split_into_chunks(X, y, n_chunks=4, random_state=0)
        for Xc, yc in chunks:
            trainer.fit_chunk(Xc, yc)
            trainer.score_chunk(X, y)
        history = trainer.get_metric_history("eval_acc")
        assert len(history) == 4
        assert all(0 <= v <= 1 for v in history)

    def test_log_to_csv(self, binary_data, tmp_path):
        trainer, X, y = self._make_trainer(binary_data)
        trainer.fit_chunk(X, y)
        fpath = tmp_path / "log.csv"
        trainer.log_to_csv(str(fpath))
        assert fpath.exists()
        content = fpath.read_text()
        assert "chunk" in content


# ═════════════════════════════════════════════════════════════════════════════
# IO utilities
# ═════════════════════════════════════════════════════════════════════════════

class TestIO:

    def test_split_into_chunks_count(self, binary_data):
        X, y   = binary_data
        chunks = split_into_chunks(X, y, n_chunks=5, random_state=0)
        assert len(chunks) == 5
        assert sum(len(yc) for _, yc in chunks) == 200

    def test_train_test_split_sizes(self, binary_data):
        X, y = binary_data
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                               random_state=0)
        assert len(Xtr) == 160
        assert len(Xte) == 40

    def test_make_dataset_shapes(self):
        X, y = make_classification_dataset(
            n_samples=300, n_features=8,
            n_classes=3, random_state=0,
        )
        assert X.shape == (300, 8)
        assert y.shape == (300,)
        assert set(y).issubset({0, 1, 2})

    def test_csv_roundtrip(self, tmp_path):
        X       = np.array([[1.0, 2.0], [3.0, 4.0]])
        headers = ["a", "b"]
        fpath   = tmp_path / "test.csv"
        write_csv(fpath, X, headers=headers)
        h2, X2  = read_csv(fpath)
        assert h2 == headers
        np.testing.assert_allclose(X2, X, atol=1e-5)

    def test_chunk_no_data_loss(self, binary_data):
        X, y   = binary_data
        chunks = split_into_chunks(X, y, n_chunks=7,
                                   shuffle=False, random_state=0)
        assert sum(len(yc) for _, yc in chunks) == len(y)


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_single_class_tree(self):
        X    = np.random.randn(20, 2)
        y    = np.zeros(20, dtype=int)
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)
        assert np.all(tree.predict(X) == 0)

    def test_zero_variance_scaler_chunk(self):
        s  = StandardScaler()
        s.partial_fit(np.ones((10, 3)))
        Xt = s.transform(np.ones((5, 3)))
        assert np.all(np.isfinite(Xt))

    def test_streaming_accuracy_empty(self):
        m = Accuracy()
        assert m.compute() == 0.0

    def test_pipeline_score_after_partial_fit(self, binary_data):
        X, y   = binary_data
        pipe   = Pipeline([
            ("scaler", StandardScaler()),
            ("tree",   DecisionTreeClassifier(max_depth=3, random_state=0)),
        ])
        chunks = split_into_chunks(X, y, n_chunks=4, random_state=0)
        for Xc, yc in chunks:
            pipe.partial_fit(Xc, yc)
        assert 0 < pipe.score(X, y) <= 1.0

    def test_ensemble_adaboost_multiclass(self, multiclass_data):
        X, y = multiclass_data
        ens  = EnsembleClassifier(method="adaboost", n_estimators=3,
                                   random_state=0)
        ens.fit(X, y)
        assert set(ens.predict(X)).issubset({0, 1, 2})

    def test_imputer_all_nan_column(self):
        X      = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        imp    = Imputer(strategy="constant", fill_value=0.0)
        Xt     = imp.fit_transform(X)
        assert np.all(Xt[:, 0] == 0.0)