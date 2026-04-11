# src/evaluate.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from config import CLASS_NAMES


def evaluate(model, val_ds):
    """Prints accuracy and plots confusion matrix."""
    loss, accuracy = model.evaluate(val_ds)
    print(f"\n Validation Accuracy: {accuracy * 100:.2f}%")
    print(f" Validation Loss:     {loss:.4f}")
    return accuracy, loss


def plot_training(history):
    """Plots accuracy and loss curves."""
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc,     label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss,     label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
    print("Training curves saved to training_curves.png")


def plot_confusion_matrix(model, val_ds):
    """Generates and plots the confusion matrix."""
    y_true, y_pred = [], []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                       labels=list(range(len(CLASS_NAMES)))))