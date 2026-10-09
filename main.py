import argparse

from config import IMG_SIZE
from src.data_loader import load_data
from src.evaluate import evaluate, plot_training
from src.model import build_model
from src.train import best_epoch, train


def main():
    parser = argparse.ArgumentParser(description="Train the flower classifier")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Train on train+validation using the epoch counts in config.py",
    )
    args = parser.parse_args()
    mode = "final" if args.final else "tuning"

    print("=" * 45)
    print(f"     Flower Species Classifier ({mode} run)")
    print("=" * 45)

    print("\n Loading dataset...")
    train_ds, val_ds, test_ds, class_names = load_data(final=args.final)

    print("\n Building model...")
    model, base_model = build_model()
    model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    model.summary()

    history1, history2 = train(model, base_model, train_ds, val_ds)

    run_info = {"mode": mode}
    if not args.final:
        val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
        run_info.update(
            {
                "best_val_accuracy": round(max(val_acc), 4),
                "best_phase1_epoch": best_epoch(history1),
                "best_phase2_epoch": best_epoch(history2),
            }
        )

        plot_training(history1)
    evaluate(model, test_ds, class_names, run_info)

    print("\n Done!")


if __name__ == "__main__":
    main()
