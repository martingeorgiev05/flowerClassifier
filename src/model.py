import tensorflow as tf
from config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

def build_model():
    """Builds and returns a MobileNetV2-based transfer learning model."""

    #Load pretrained base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape = (IMG_SIZE, IMG_SIZE, 3),
        include_top = False,
        weights = 'imagenet'
    )
    base_model.trainable = False # Freeze pretrained weights

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255), # Normalize pixels to [0,1]
        base_model, # Pretrained feature extractor
        tf.keras.layers.GlobalAveragePooling2D(), # Flatten feature maps
        tf.keras.layers.Dropout(0.2), # Reduce overfitting
        tf.keras.layers.Dense(NUM_CLASSES, activation = 'softmax') # Output layer
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model