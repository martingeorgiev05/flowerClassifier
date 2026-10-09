import numpy as np
import tensorflow as tf

from config import EPOCHS, FINAL_PHASE1_EPOCHS, FINAL_PHASE2_EPOCHS, MODEL_SAVE_PATH


def best_epoch(history):
    """1-based epoch with the lowest val_loss."""
    return int(np.argmin(history.history["val_loss"])) + 1


def _callbacks(final):
    if final:
        # No validation set: only lower the LR when the training loss stalls
        return [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1
            )
        ]

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1
        ),
    ]


def train(model, base_model, train_ds, val_ds=None):
    """Two-phase training.Pass val_ds=None for a final run on train+validation."""

    final = val_ds is None
    phase1_epochs = FINAL_PHASE1_EPOCHS if final else EPOCHS
    phase2_epochs = FINAL_PHASE2_EPOCHS if final else EPOCHS // 2
    callbacks = _callbacks(final)

    print("\n Phase 1: Training top layers....\n")
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=phase1_epochs, callbacks=callbacks
    )

    print("\n Phase 2: Fine-tuning base model....")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=callbacks,
    )

    if final:
        model.save(MODEL_SAVE_PATH)
    else:
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    print(f"\n Model saved to: {MODEL_SAVE_PATH}")
    return model, history1, history2
