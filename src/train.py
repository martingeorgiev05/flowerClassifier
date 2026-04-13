import tensorflow as tf
from config import EPOCHS, MODEL_SAVE_PATH

def train(model, base_model, train_ds, val_ds):

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print("\n Phase 1: Training top layers...\n")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    print("\n Phase 2: Fine-tuning base model...\n")

    base_model.trainable = True

    # Freeze lower layers, train top layers only
    for layer in base_model.layers[:-30]:
        layer.trainable = False


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS // 2,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    print(f"\n Model saved to: {MODEL_SAVE_PATH}")

    return history1, history2