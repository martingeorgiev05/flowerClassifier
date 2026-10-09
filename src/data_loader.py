import tensorflow as tf
import tensorflow_datasets as tfds

from config import BATCH_SIZE, IMG_SIZE

AUTOTUNE = tf.data.AUTOTUNE


def _preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


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
        .cache()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    if val_ds is not None:
        val_ds = (
            val_ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
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
