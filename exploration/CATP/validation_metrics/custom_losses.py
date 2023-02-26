import tensorflow as tf


def non_zero_mape(y_true, y_pred):
    t = y_true[y_true > 10]
    p = y_pred[y_true > 10]

    squared_difference = tf.abs((t - p) / t)
    return tf.reduce_mean(squared_difference, axis=-1)
