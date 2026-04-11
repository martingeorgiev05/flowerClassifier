# main.py
from src.data_loader import load_data
from src.model import build_model
from src.train import train
from src.evaluate import evaluate, plot_training, plot_confusion_matrix
import sys

def main():
    print("=" * 45)
    print("  Flower Species Classifier")
    print("=" * 45)

    # Step 1 — Load data
    print("\n📂 Loading dataset...")
    train_ds, val_ds = load_data()

    # Step 2 — Build model
    print("\n🏗️  Building model...")
    model = build_model()
    model.summary()

    # Step 3 — Train
    history = train(model, train_ds, val_ds)

    # Step 4 — Evaluate
    evaluate(model, val_ds)
    plot_training(history)
    plot_confusion_matrix(model, val_ds)

    print("\n Done! Your model is trained and ready.")


if __name__ == "__main__":
    main()