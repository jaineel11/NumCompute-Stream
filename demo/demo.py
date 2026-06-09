"""
demo/stream_demo.py
===================
End-to-end streaming ML demo using a customer churn dataset.

What this demo shows
--------------------
1.  Load customer_churn.csv using io.py
2.  Split into chunks to simulate a streaming data source
3.  Build three Pipeline instances (DecisionTree, RandomForest, AdaBoost)
4.  Train each pipeline incrementally via StreamTrainer.fit_chunk()
5.  Evaluate after every chunk via StreamTrainer.score_chunk()
6.  Log and visualise metrics over time using visualise.py
7.  Print the StreamTrainer summary table
8.  Save training logs to CSV
9.  Show per-chunk descriptive statistics via update_stats()

Run
---
    python demo/stream_demo.py
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
    split_into_chunks,
    train_test_split,
)
from numcompute_stream.metrics import (
    Accuracy,
    confusion_matrix as batch_cm,
)
from numcompute_stream.stats import (
    update_stats,
    reset_stats,
)

import matplotlib.pyplot as plt


# Config

N_CHUNKS    = 12
RANDOM_SEED = 42
DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "outputs")


# Helpers

def banner(text):
    width = 62
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def section(text):
    print(f"\n── {text}")


# Step 1 – Load dataset

def load_dataset():
    section("Step 1: Load customer churn dataset from CSV")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
        )

    headers, data = read_csv(DATA_PATH)
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    feature_names = headers[:-1]
    classes, counts = np.unique(y, return_counts=True)

    print(f"  File     : {DATA_PATH}")
    print(f"  Samples  : {X.shape[0]}")
    print(f"  Features : {X.shape[1]}  {feature_names}")
    print(f"  Classes  : churn=0 (stay) → {counts[0]} customers")
    print(f"             churn=1 (left) → {counts[1]} customers")
    print(f"  Churn rate: {counts[1]/len(y)*100:.1f}%")

    return X, y, feature_names


# Step 2 – Train/test split + chunks

def prepare_chunks(X, y):
    section("Step 2: Split into train/test and create streaming chunks")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED,
    )
    chunks = split_into_chunks(
        X_tr, y_tr, n_chunks=N_CHUNKS,
        shuffle=True, random_state=RANDOM_SEED,
    )

    print(f"  Train    : {len(y_tr)} customers  →  {N_CHUNKS} chunks of ~{len(chunks[0][1])} each")
    print(f"  Test     : {len(y_te)} customers (held out, never seen during training)")

    return X_tr, X_te, y_tr, y_te, chunks


# Step 3 – Build pipelines

def build_pipelines():
    section("Step 3: Build three streaming pipelines")

    pipelines = {
        "DecisionTree": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  DecisionTreeClassifier(
                max_depth=6, criterion="gini", random_state=0,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  EnsembleClassifier(
                method="random_forest", n_estimators=8,
                max_depth=5, random_state=0,
            )),
        ]),
        "AdaBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  EnsembleClassifier(
                method="adaboost", n_estimators=8,
                random_state=0,
            )),
        ]),
    }

    print("  Each pipeline: StandardScaler → model")
    for name in pipelines:
        print(f"  {name}")

    return pipelines


# Step 4 – Stream training

def stream_train(pipelines, chunks, X_te, y_te):
    section("Step 4: Incremental training via StreamTrainer")

    classes  = np.unique(np.concatenate([yc for _, yc in chunks]))
    trainers = {
        name: StreamTrainer(pipeline=pipe, verbose=False)
        for name, pipe in pipelines.items()
    }

    print(f"\n  {'Chunk':>5}  {'Customers':>10}  " +
          "  ".join(f"{n:>14}" for n in pipelines))
    print("  " + "─" * (18 + 16 * len(pipelines)))

    for idx, (Xc, yc) in enumerate(chunks):
        row = f"  {idx+1:>5}  {len(yc):>10}  "
        for name, trainer in trainers.items():
            trainer.fit_chunk(Xc, yc)
            acc = trainer.score_chunk(X_te, y_te)
            row += f"{acc:>14.4f}  "
        print(row)

    print()
    return trainers


# Step 5 – Final evaluation

def final_eval(pipelines, X_te, y_te):
    section("Step 5: Final evaluation on held-out test set")

    best_name = None
    best_acc  = -1.0
    final_accs = {}

    for name, pipe in pipelines.items():
        acc = pipe.score(X_te, y_te)
        preds = pipe.predict(X_te)
        n_correct = int(np.sum(preds == y_te))
        final_accs[name] = acc
        print(f"  {name:<14}: accuracy={acc:.4f}  ({n_correct}/{len(y_te)} correct)")
        if acc > best_acc:
            best_acc  = acc
            best_name = name

    print(f"\n  Best model: {best_name}  (acc={best_acc:.4f})")
    return final_accs, best_name


# Step 6 – Visualisations

def make_visualisations(pipelines, trainers, X_te, y_te, best_name, feature_names):
    section("Step 6: Saving visualisations to demo/outputs/")

    os.makedirs(OUT_DIR, exist_ok=True)

    # accuracy over time per model
    for name, trainer in trainers.items():
        history = trainer.get_metric_history("eval_acc")
        fig = visualise.plot_metric_over_time(
            history,
            title  = f"{name} — Test Accuracy per Chunk (Customer Churn)",
            ylabel = "Accuracy",
            xlabel = "Chunk",
            smoothing = 3,
            save_path = os.path.join(OUT_DIR, f"{name.lower()}_accuracy.png"),
        )
        plt.close(fig)

    print("  Saved per-model accuracy over time plots")

    # compare_models
    names    = list(trainers.keys())
    fig = visualise.compare_models(
        trainers[names[0]].get_metric_history("eval_acc"),
        trainers[names[1]].get_metric_history("eval_acc"),
        labels    = [names[0], names[1]],
        title     = f"Model Comparison: {names[0]} vs {names[1]}",
        ylabel    = "Accuracy",
        save_path = os.path.join(OUT_DIR, "model_comparison.png"),
    )
    plt.close(fig)
    print(f"  Saved compare_models: {names[0]} vs {names[1]}")

    # predictions vs ground truth
    best_pipe = pipelines[best_name]
    y_pred    = best_pipe.predict(X_te)
    fig = visualise.plot_predictions_vs_ground_truth(
        y_te,
        y_pred,
        title     = f"{best_name} — Churn Predictions vs Actual (Test Set)",
        save_path = os.path.join(OUT_DIR, "predictions_vs_truth.png"),
    )
    plt.close(fig)
    print("  Saved predictions vs ground truth")

    # confusion matrix
    mat, cls = batch_cm(y_te, y_pred)
    fig = visualise.plot_confusion_matrix(
        mat, ["Stay (0)", "Churn (1)"],
        title     = f"{best_name} — Confusion Matrix",
        normalise = False,
        save_path = os.path.join(OUT_DIR, "confusion_matrix.png"),
    )
    plt.close(fig)
    print("  Saved confusion matrix")

    # learning curve
    best_trainer = trainers[best_name]
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
        title          = f"{best_name} — Learning Curve (Customers Seen vs Accuracy)",
        save_path      = os.path.join(OUT_DIR, "learning_curve.png"),
    )
    plt.close(fig)
    print("  Saved learning curve")

    print(f"\n  All plots saved to {OUT_DIR}/")


# Step 7 – StreamTrainer summary

def print_summary(trainers, best_name):
    section("Step 7: StreamTrainer summary table")
    print(f"\n  Showing log for best model: {best_name}")
    trainers[best_name].summary()


# Step 8 – Save logs

def save_logs(trainers):
    section("Step 8: Save training logs to CSV")
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, trainer in trainers.items():
        path = os.path.join(OUT_DIR, f"{name.lower()}_log.csv")
        trainer.log_to_csv(path)


# Step 9 – Streaming stats demo

def streaming_stats_demo(chunks, feature_names):
    section("Step 9: Running descriptive stats on monthly_charges via update_stats()")

    # monthly_charges is column index 2
    col_idx = 2
    col_name = feature_names[col_idx]

    reset_stats()
    print(f"\n  Tracking column: {col_name}")
    print(f"\n  {'Chunk':>6}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}  {'N':>6}")
    print("  " + "─" * 56)

    for idx, (Xc, _) in enumerate(chunks):
        col_data = Xc[:, col_idx].reshape(-1, 1)
        stats    = update_stats(col_data)
        mean_val = stats["mean"][0] if hasattr(stats["mean"], "__len__") else stats["mean"]
        std_val  = stats["std"][0]  if hasattr(stats["std"],  "__len__") else stats["std"]
        min_val  = stats["min"][0]  if hasattr(stats["min"],  "__len__") else stats["min"]
        max_val  = stats["max"][0]  if hasattr(stats["max"],  "__len__") else stats["max"]
        print(
            f"  {idx+1:>6}  {mean_val:>10.2f}  {std_val:>10.2f}  "
            f"{min_val:>10.2f}  {max_val:>10.2f}  {stats['n']:>6}"
        )


# Main

def main():
    banner("numcompute_stream — Customer Churn Streaming Demo")
    print("  Pure NumPy + matplotlib. No scikit-learn.")
    print(f"  Output directory: {OUT_DIR}")

    X, y, feature_names = load_dataset()
    X_tr, X_te, y_tr, y_te, chunks = prepare_chunks(X, y)
    pipelines = build_pipelines()
    trainers  = stream_train(pipelines, chunks, X_te, y_te)
    final_accs, best_name = final_eval(pipelines, X_te, y_te)
    make_visualisations(pipelines, trainers, X_te, y_te, best_name, feature_names)
    print_summary(trainers, best_name)
    save_logs(trainers)
    streaming_stats_demo(chunks, feature_names)

    banner("Demo Complete")
    print(f"  Dataset      : Customer Churn ({len(y)} customers)")
    print(f"  Features     : {feature_names}")
    print(f"  Chunks       : {N_CHUNKS}")
    print(f"  Train        : {len(y_tr)} customers")
    print(f"  Test         : {len(y_te)} customers")
    print(f"  Best model   : {best_name}  (acc={final_accs[best_name]:.4f})")
    print(f"  Plots saved  : {OUT_DIR}/")
    print()


if __name__ == "__main__":
    main()