"""
demo/stream_demo.py
===================
End-to-end streaming ML demo for numcompute_stream.

What this demo shows
--------------------
1.  Load a dataset from a CSV file using io.py
2.  Split into chunks to simulate a streaming data source
3.  Build three Pipeline instances (DecisionTree, RandomForest, AdaBoost)
4.  Train each pipeline incrementally via StreamTrainer.fit_chunk()
5.  Evaluate after every chunk via StreamTrainer.score_chunk()
6.  Log and visualise key metrics over time using visualise.py:
        - Accuracy over chunks (per model)
        - Model comparison plot
        - Predictions vs ground truth on the final chunk
        - Confusion matrix after full stream
        - Learning curve (samples seen vs accuracy)
7.  Print the StreamTrainer summary table
8.  Save the training log to CSV

Run
---
    python demo/stream_demo.py

All output plots are saved to demo/outputs/.
The training log CSV is saved to demo/outputs/training_log.csv.
No external ML libraries required — NumPy + matplotlib only.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream import (
    Pipeline,
    StandardScaler,
    DecisionTreeClassifier,
    EnsembleClassifier,
    StreamTrainer,
    visualise,
)
from numcompute_stream.io import (
    read_csv,
    write_csv,
    split_into_chunks,
    train_test_split,
    make_classification_dataset,
)
from numcompute_stream.metrics import (
    Accuracy,
    ConfusionMatrix,
    confusion_matrix as batch_cm,
)
from numcompute_stream.stats import (
    StreamingStats,
    update_stats,
    reset_stats,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

N_CHUNKS    = 10
N_SAMPLES   = 800
N_FEATURES  = 8
N_CLASSES   = 3
RANDOM_SEED = 42
OUT_DIR     = os.path.join(os.path.dirname(__file__), "outputs")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def banner(text):
    width = 60
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def section(text):
    print(f"\n── {text}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Generate and load dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    section("Step 1: Load dataset from CSV")

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # generate a synthetic dataset and save it if not present
    if not os.path.exists(DATA_PATH):
        print("  Generating synthetic dataset...")
        X, y = make_classification_dataset(
            n_samples   = N_SAMPLES,
            n_features  = N_FEATURES,
            n_classes   = N_CLASSES,
            n_informative = 6,
            noise       = 0.3,
            random_state = RANDOM_SEED,
        )
        data    = np.hstack([X, y.reshape(-1, 1)])
        headers = [f"feature_{i}" for i in range(N_FEATURES)] + ["label"]
        write_csv(DATA_PATH, data, headers=headers)
        print(f"  Saved to {DATA_PATH}")

    headers, data = read_csv(DATA_PATH)
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    print(f"  Loaded  : {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes")
    print(f"  Headers : {headers}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Train / test split + chunk creation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_chunks(X, y):
    section("Step 2: Split into train/test and create streaming chunks")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED,
    )
    chunks = split_into_chunks(
        X_tr, y_tr, n_chunks=N_CHUNKS,
        shuffle=True, random_state=RANDOM_SEED,
    )

    print(f"  Train   : {len(y_tr)} samples → {N_CHUNKS} chunks "
          f"of ~{len(chunks[0][1])} each")
    print(f"  Test    : {len(y_te)} samples")
    return X_tr, X_te, y_tr, y_te, chunks


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Build pipelines
# ─────────────────────────────────────────────────────────────────────────────

def build_pipelines():
    section("Step 3: Build streaming pipelines")

    pipelines = {
        "DecisionTree": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeClassifier(
                max_depth=5, criterion="gini", random_state=0,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  EnsembleClassifier(
                method="random_forest", n_estimators=5,
                max_depth=4, random_state=0,
            )),
        ]),
        "AdaBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  EnsembleClassifier(
                method="adaboost", n_estimators=5,
                random_state=0,
            )),
        ]),
    }

    for name, pipe in pipelines.items():
        print(f"  {name:<14}: {pipe}")

    return pipelines


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Stream training with StreamTrainer
# ─────────────────────────────────────────────────────────────────────────────

def stream_train(pipelines, chunks, X_te, y_te):
    section("Step 4: Incremental training via StreamTrainer")

    classes  = np.unique(
        np.concatenate([yc for _, yc in chunks])
    )
    trainers = {
        name: StreamTrainer(pipeline=pipe, verbose=False)
        for name, pipe in pipelines.items()
    }

    # streaming metrics for live tracking
    acc_meters = {name: Accuracy() for name in pipelines}

    print(f"\n  {'Chunk':>5}  " +
          "  ".join(f"{n:>14}" for n in pipelines))
    print("  " + "─" * (7 + 16 * len(pipelines)))

    for idx, (Xc, yc) in enumerate(chunks):
        row = f"  {idx+1:>5}  "
        for name, trainer in trainers.items():
            trainer.fit_chunk(Xc, yc)
            acc = trainer.score_chunk(X_te, y_te)
            acc_meters[name].update(y_te, pipelines[name].predict(X_te))
            row += f"{acc:>14.4f}  "
        print(row)

    print()
    return trainers, acc_meters


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Final evaluation
# ─────────────────────────────────────────────────────────────────────────────

def final_eval(pipelines, trainers, X_te, y_te):
    section("Step 5: Final evaluation on held-out test set")

    best_name = None
    best_acc  = -1.0
    final_accs = {}

    for name, pipe in pipelines.items():
        acc = pipe.score(X_te, y_te)
        final_accs[name] = acc
        print(f"  {name:<14}: accuracy = {acc:.4f}")
        if acc > best_acc:
            best_acc  = acc
            best_name = name

    print(f"\n  Best model: {best_name}  (acc={best_acc:.4f})")
    return final_accs, best_name


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def make_visualisations(pipelines, trainers, X_te, y_te, best_name):
    section("Step 6: Saving visualisations")

    os.makedirs(OUT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    # ── 6a. Accuracy over time for each model ────────────────────────────────
    for name, trainer in trainers.items():
        history = trainer.get_metric_history("eval_acc")
        fig = visualise.plot_metric_over_time(
            history,
            title  = f"{name} – Test Accuracy per Chunk",
            ylabel = "Accuracy",
            xlabel = "Chunk",
            smoothing = 3,
            save_path = os.path.join(OUT_DIR, f"{name.lower()}_accuracy.png"),
        )
        plt.close(fig)
    print("  Saved per-model accuracy plots")

    # ── 6b. compare_models (spec required) ───────────────────────────────────
    names    = list(trainers.keys())
    history0 = trainers[names[0]].get_metric_history("eval_acc")
    history1 = trainers[names[1]].get_metric_history("eval_acc")

    fig = visualise.compare_models(
        history0,
        history1,
        labels    = [names[0], names[1]],
        title     = f"Model Comparison: {names[0]} vs {names[1]}",
        ylabel    = "Accuracy",
        save_path = os.path.join(OUT_DIR, "model_comparison.png"),
    )
    plt.close(fig)
    print(f"  Saved compare_models: {names[0]} vs {names[1]}")

    # also compare best vs third model if 3+ models
    if len(names) >= 3:
        history2 = trainers[names[2]].get_metric_history("eval_acc")
        fig = visualise.compare_models(
            history1,
            history2,
            labels    = [names[1], names[2]],
            title     = f"Model Comparison: {names[1]} vs {names[2]}",
            ylabel    = "Accuracy",
            save_path = os.path.join(OUT_DIR, "model_comparison_2.png"),
        )
        plt.close(fig)

    # ── 6c. plot_predictions_vs_ground_truth (spec required) ─────────────────
    best_pipe = pipelines[best_name]
    y_pred    = best_pipe.predict(X_te)
    fig = visualise.plot_predictions_vs_ground_truth(
        y_te,
        y_pred,
        title     = f"{best_name} – Predictions vs Ground Truth",
        save_path = os.path.join(OUT_DIR, "predictions_vs_truth.png"),
    )
    plt.close(fig)
    print("  Saved plot_predictions_vs_ground_truth")

    # ── 6d. Confusion matrix ──────────────────────────────────────────────────
    mat, cls = batch_cm(y_te, y_pred)
    fig = visualise.plot_confusion_matrix(
        mat, cls,
        title     = f"{best_name} – Confusion Matrix",
        normalise = False,
        save_path = os.path.join(OUT_DIR, "confusion_matrix.png"),
    )
    plt.close(fig)
    print("  Saved confusion matrix")

    # ── 6e. Learning curve ────────────────────────────────────────────────────
    best_trainer  = trainers[best_name]
    eval_history  = best_trainer.get_metric_history("eval_acc")
    train_history = best_trainer.get_metric_history("train_acc")
    n_samples_log = [
        entry["n_total"]
        for entry in best_trainer.log_
        if entry.get("eval_acc") is not None
    ]

    fig = visualise.plot_learning_curve(
        n_samples_list = n_samples_log,
        train_scores   = train_history[:len(n_samples_log)],
        val_scores     = eval_history,
        metric_name    = "Accuracy",
        title          = f"{best_name} – Learning Curve",
        save_path      = os.path.join(OUT_DIR, "learning_curve.png"),
    )
    plt.close(fig)
    print("  Saved learning curve")

    # ── 6f. plot_metric_over_time for cumulative accuracy ────────────────────
    cum_history = [
        entry["cumulative_acc"]
        for entry in best_trainer.log_
        if entry.get("cumulative_acc") is not None
    ]
    fig = visualise.plot_metric_over_time(
        cum_history,
        title     = f"{best_name} – Cumulative Accuracy",
        ylabel    = "Cumulative Accuracy",
        xlabel    = "Chunk",
        save_path = os.path.join(OUT_DIR, "cumulative_accuracy.png"),
    )
    plt.close(fig)
    print("  Saved cumulative accuracy plot")

    print(f"\n  All visualisations saved to {OUT_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – StreamTrainer summary + log
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(trainers, best_name):
    section("Step 7: StreamTrainer summary table")
    print(f"\n  Best model: {best_name}")
    trainers[best_name].summary()


def save_logs(trainers):
    section("Step 8: Save training logs to CSV")
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, trainer in trainers.items():
        path = os.path.join(OUT_DIR, f"{name.lower()}_log.csv")
        trainer.log_to_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 – Streaming stats demo
# ─────────────────────────────────────────────────────────────────────────────

def streaming_stats_demo(chunks):
    section("Step 9: Per-chunk descriptive statistics via update_stats()")

    reset_stats()
    print(f"\n  {'Chunk':>6}  {'Mean[0]':>10}  {'Std[0]':>10}  {'N':>8}")
    print("  " + "─" * 40)

    for idx, (Xc, _) in enumerate(chunks):
        stats = update_stats(Xc)
        mean0 = stats["mean"][0] if hasattr(stats["mean"], "__len__") \
                else stats["mean"]
        std0  = stats["std"][0]  if hasattr(stats["std"],  "__len__") \
                else stats["std"]
        print(f"  {idx+1:>6}  {mean0:>10.4f}  {std0:>10.4f}  "
              f"{stats['n']:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    banner("numcompute_stream — Streaming ML Demo")
    print("  Pure NumPy + matplotlib. No scikit-learn.")
    print(f"  Output directory: {OUT_DIR}")

    # 1. data
    X, y = load_dataset()

    # 2. splits + chunks
    X_tr, X_te, y_tr, y_te, chunks = prepare_chunks(X, y)

    # 3. pipelines
    pipelines = build_pipelines()

    # 4. stream training
    trainers, acc_meters = stream_train(
        pipelines, chunks, X_te, y_te,
    )

    # 5. final eval
    final_accs, best_name = final_eval(
        pipelines, trainers, X_te, y_te,
    )

    # 6. visualisations
    make_visualisations(
        pipelines, trainers, X_te, y_te, best_name,
    )

    # 7. summary table
    print_summary(trainers, best_name)

    # 8. save logs
    save_logs(trainers)

    # 9. stats demo
    streaming_stats_demo(chunks)

    # ── final summary ─────────────────────────────────────────────────────────
    banner("Demo Complete")
    print(f"  Chunks trained : {N_CHUNKS}")
    print(f"  Train samples  : {len(y_tr)}")
    print(f"  Test samples   : {len(y_te)}")
    print(f"  Best model     : {best_name}  "
          f"(acc={final_accs[best_name]:.4f})")
    print(f"  Plots saved to : {OUT_DIR}/")
    print()


if __name__ == "__main__":
    main()