from config import IMG_SIZE
from src.data_loader import load_data
from src.model import build_model
from src.train import train
from src.evaluate import evaluate, plot_training, plot_confusion_matrix

def main():
    print("=" * 45)
    print("     Flower Species Classifier")
    print("=" * 45)

    print("\n Loading dataset...")
    train_ds, val_ds = load_data()

    print("\n Building model...")
    model, base_model = build_model() 
    model.build(input_shape=(None, IMG_SIZE, IMG_SIZE, 3))
    model.summary()

    history1, history2 = train(model, base_model, train_ds, val_ds)  

    evaluate(model, val_ds)
    plot_training(history1) 
    plot_confusion_matrix(model, val_ds)

    print("\n Done!")

if __name__ == "__main__":
    main()