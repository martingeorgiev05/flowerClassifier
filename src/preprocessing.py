import tensorflow as tf

from config import IMG_SIZE


def preprocess_image(image):
    """Resizes the shorter side to IMG_SIZE (keeping proportions), then center-crops to IMG_SIZE xIMG_SIZE."""

    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    scale = IMG_SIZE / tf.reduce_min(shape)
    new_size = tf.cast(tf.math.ceil(shape * scale), tf.int32)
    image = tf.image.resize(image, new_size, antialias=True)
    return tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)
