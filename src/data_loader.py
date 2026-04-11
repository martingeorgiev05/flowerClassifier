import tensorflow as tf
import pathlib
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, SEED

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

def download_data():
    """Downloads the flower dataset if not already present."""
    data_dir = tf.keras.utils.get_file(
        'flower_photos',
        origin=DATASET_URL,
        untar=True
    )
    data_dir = pathlib.Path(data_dir)

    # Fix nested folder issue
    nested = data_dir / "flower_photos"
    if nested.exists():
        data_dir = nested

    print(f"\n📁 Dataset path: {data_dir}")
    print(f"📂 Subfolders found: {[f.name for f in data_dir.iterdir() if f.is_dir()]}")

    return data_dir


def load_data():
    """Loads and returns (train_ds, val_ds) datasets."""
    data_dir = download_data()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds