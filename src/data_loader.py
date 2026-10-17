import tensorflow as tf
import tensorflow_datasets as tfds

from config import BATCH_SIZE, MIXUP_ALPHA, NUM_CLASSES
from src.preprocessing import preprocess_image

AUTOTUNE = tf.data.AUTOTUNE


def _sample_beta(size, alpha):
    """Beta(alpha, alpha) samples via two Gamma draws"""
    g1 = tf.random.gamma([size], alpha)
    g2 = tf.random.gamma([size], alpha)
    return tf.math.divide_no_nan(g1, g1 + g2)


def _mixup(images, labels):
    """Blends each image and its one-hot label with another random image from the same batch."""
    batch_size = tf.shape(images)[0]
    lam = _sample_beta(batch_size, MIXUP_ALPHA)
    partner = tf.random.shuffle(tf.range(batch_size))
    lam_images = tf.reshape(lam, [-1, 1, 1, 1])
    lam_labels = tf.reshape(lam, [-1, 1])
    images = lam_images * images + (1 - lam_images) * tf.gather(images, partner)
    labels = lam_labels * labels + (1 - lam_labels) * tf.gather(labels, partner)
    return images, labels


def _preprocess(image, label):
    return preprocess_image(image), label


def _one_hot(image, label):
    return image, tf.one_hot(label, NUM_CLASSES)


def load_data(final=False):
    """
    Tuning run(final=False): train on `train`, validate on `validation`.
    Final run(final=True): train on `train` + `validation`, no validation set.
    `test` is only ever used for the reported metrics.
    """
    (train_ds, val_ds, test_ds), info = tfds.load(
        "oxford_flowers102",
        split=["train", "validation", "test"],
        as_supervised=True,
        with_info=True,
    )
    class_names = info.features["label"].names

    if final:
        train_ds = train_ds.concatenate(val_ds)
        val_ds = None

    train_ds = (
        train_ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
        .map(_one_hot, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(2048)
        .batch(BATCH_SIZE)
    )
    if MIXUP_ALPHA > 0:
        train_ds = train_ds.map(_mixup, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    if val_ds is not None:
        val_ds = (
            val_ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
            .map(_one_hot, num_parallel_calls=AUTOTUNE)
            .cache()
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )

    test_ds = (
        test_ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, class_names
