# src/data_loader.py
import tensorflow as tf
import tensorflow_datasets as tfds
from config import IMG_SIZE, BATCH_SIZE

def load_data():
    (train_ds, val_ds), info = tfds.load(
        'oxford_flowers102',
        split=['train', 'validation'],
        as_supervised=True,
        with_info=True
    )

    def preprocess(image, label):
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(preprocess).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, val_ds