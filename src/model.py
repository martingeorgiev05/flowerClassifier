import tensorflow as tf
from config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

def build_model():

    # Data augmentation (NEW)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1),
    ])

    # Load pretrained base
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False  

    model = tf.keras.Sequential([
        data_augmentation, 

        tf.keras.layers.Lambda(
            tf.keras.applications.efficientnet.preprocess_input
        ),

        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),   
        tf.keras.layers.Dropout(0.4),                    
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),                    
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model