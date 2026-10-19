import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import RESULTS_DIR, USE_TTA
from src.predict import predict_probs


def predict_dataset(model, ds, tta=USE_TTA):
    """Runs the model over a dataset once.Returns(true labels, class probabilities)."""
    y_true, y_prob = [], []
    for images, labels in tqdm(ds, desc="Predicting", unit="batch"):
        y_prob.append(predict_probs(model, images, tta))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def most_confused(cm, class_names, n=10):
    """The n largest off-diagonal cells of the confusion matrix."""
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    flat_idx = np.argsort(off_diag, axis=None)[::-1][:n]
    rows, cols = np.unravel_index(flat_idx, off_diag.shape)
    return [
        {
            "true": class_names[t],
            "predicted": class_names[p],
            "count": int(off_diag[t, p]),
        }
        for t, p in zip(rows, cols, strict=True)
        if off_diag[t, p] > 0
    ]


def evaluate(model, test_ds, class_names, run_info, tta=USE_TTA):
    """Predicts the test split,then reports and saves the results."""
    y_true, y_prob = predict_dataset(model, test_ds, tta)
    return report(y_true, y_prob, class_names, {**run_info, "tta": tta})


def report(y_true, y_prob, class_names, run_info, out_dir=RESULTS_DIR, split="test"):
    """Top-1/top-5 accuracy, most confused paris, classification report and confusion matrix."""
    os.makedirs(out_dir, exist_ok=True)
    mode = run_info["mode"]

    y_pred = y_prob.argmax(axis=1)
    top5 = np.argsort(y_prob, axis=1)[:, -5:]

    top1_acc = float(np.mean(y_pred == y_true))
    top5_acc = float(np.mean(np.any(top5 == y_true[:, None], axis=1)))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    confused = most_confused(cm, class_names)

    print(f"\n {split.capitalize()} images: {len(y_true)} ({mode})")
    print(f" Top-1 accuracy: {top1_acc * 100:.2f}%")
    print(f" Top-5 accuracy: {top5_acc * 100:.2f}%")
    print("\n Most confused pairs(true -> predicted): ")
    for c in confused:
        print(f"     {c['true']:<28} -> {c['predicted']:<28} {c['count']}")

    metrics = {
        **run_info,
        "split": split,
        "num_images": int(len(y_true)),
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "most_confused": confused,
    }
    with open(os.path.join(out_dir, f"metrics_{mode}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=range(len(class_names)),
        target_names=class_names,
        digits=3,
    )
    with open(os.path.join(out_dir, f"classification_report_{mode}.txt"), "w") as f:
        f.write(report_text)

    plot_confusion_matrix(cm, os.path.join(out_dir, f"confusion_matrix_{mode}.png"), split)
    print(f"\n Results saved to {out_dir}/")

    return metrics


def plot_confusion_matrix(cm, path, split="test"):
    """102x102 is too big for per-cell numbers, so plot per-class recall as colour."""

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Fraction of true class"},
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"Confusion matrix ({split} set, row-normalised)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_training(history1, history2=None):
    """Plots accuracy and loss for both phases"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    histories = [history1] if history2 is None else [history1, history2]
    curves = {
        key: sum((h.history[key] for h in histories), [])
        for key in ("accuracy", "val_accuracy", "loss", "val_loss")
    }
    epochs = range(1, len(curves["accuracy"]) + 1)
    phase2_start = len(history1.history["accuracy"]) + 0.5

    plt.figure(figsize=(12, 4))
    for i, (key, title) in enumerate(
        [("accuracy", "Accuracy"), ("loss", "Loss")], start=1
    ):
        plt.subplot(1, 2, i)
        plt.plot(epochs, curves[key], label=f"Train {title}")
        plt.plot(epochs, curves[f"val_{key}"], label=f"Val {title}")
        if history2 is not None:
            plt.axvline(
                phase2_start, color="gray", linestyle="--", label="Fine-tuning starts"
            )
        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"))
    plt.close()
    print(f"Training curves saved to {RESULTS_DIR}/training_curves.png")
