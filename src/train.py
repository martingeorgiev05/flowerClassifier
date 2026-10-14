import numpy as np
import tensorflow as tf

from config import (
    EPOCHS,
    FINAL_PHASE1_EPOCHS,
    FINAL_PHASE2_EPOCHS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_FROM,
    FINE_TUNE_LR,
    FINE_TUNE_WARMUP_EPOCHS,
    MODEL_SAVE_PATH,
)


def best_epoch(history):
    """1-based epoch with the lowest val_loss."""
    return int(np.argmin(history.history["val_loss"])) + 1


def _unfreeze_from(base_model, first_layer_name):
    """Makes every layer from `first_layer_name` onwards trainable, except BatchNormalization."""
    base_model.trainable = True
    names = [layer.name for layer in base_model.layers]
    start = names.index(first_layer_name)
    for i, layer in enumerate(base_model.layers):
        is_batch_norm = isinstance(layer, tf.keras.layers.BatchNormalization)
        layer.trainable = i >= start and not is_batch_norm


def _fine_tune_optimizer(steps_per_epoch, epochs):
    """AdamW with a linear warmup to FINE_TUNE_LR, then cosine decay towards zero."""
    warmup_steps = FINE_TUNE_WARMUP_EPOCHS * steps_per_epoch
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=max(epochs * steps_per_epoch - warmup_steps, 1),
        warmup_target=FINE_TUNE_LR,
        warmup_steps=warmup_steps,
    )
    return tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-4)


def train(model, base_model, train_ds, val_ds=None):
    """Two-phase training.Pass val_ds=None for a final run on train+validation."""

    final = val_ds is None
    phase1_epochs = FINAL_PHASE1_EPOCHS if final else EPOCHS
    phase2_epochs = FINAL_PHASE2_EPOCHS if final else FINE_TUNE_EPOCHS

    keep_best = []
    if not final:
        keep_best = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=MODEL_SAVE_PATH,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]
        reduce_lr = tf.keras.callbacks.ReduceLROnPlayteau(
            montiro="loss" if final else "val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        )

    print("\n Phase 1: Training top layers....\n")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=keep_best + [reduce_lr],
    )

    print("\n Phase 2: Fine-tuning base model....")
    _unfreeze_from(base_model, FINE_TUNE_FROM)

    model.compile(
        optimizer=_fine_tune_optimizer(int(train_ds.cardinality()), phase2_epochs),
        loss=model.loss,
        metrics=["accuracy"],
    )

    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"Trainable parameters in phase 2: {trainable:,}\n")

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase2_epochs,
        callbacks=keep_best,
    )

    if final:
        model.save(MODEL_SAVE_PATH)
    else:
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    print(f"\n Model saved to: {MODEL_SAVE_PATH}")
    return model, history1, history2
