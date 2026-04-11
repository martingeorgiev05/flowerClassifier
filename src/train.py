import tensorflow as tf
import os
from config import EPOCHS, MODEL_SAVE_PATH

def train(model, train_ds, val_ds):
    """Trains the model and saves it. Returns training history."""

    # Stop early f model stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_accuracy',
        patience = 3,
        restore_best_weights = True
    )

    # Save best model automatically during training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath =  MODEL_SAVE_PATH,
        monitor = 'val_accuracy',
        save_best_only = True,
        verbose = 1
    )

    print("\n Starting training... \n")

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = EPOCHS,
        callbacks = [early_stop, checkpoint]
    )

    print(f"\n Model saved to: {MODEL_SAVE_PATH}")
    return history