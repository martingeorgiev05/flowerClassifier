import tensorflow as tf
from config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

def build_model():
    """Builds and returns a learning model."""

    #Load pretrained base
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape = (IMG_SIZE, IMG_SIZE, 3),
        include_top = False,
        weights = 'imagenet'
    )
    base_model.trainable = False # Freeze pretrained weights

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model