import tensorflow as tf


def non_zero_mape(y_true, y_pred):
    t = y_true[y_true > 10]
    p = y_pred[y_true > 10]

    frac_abs = tf.abs((t - p) / t)
    return tf.reduce_mean(frac_abs, axis=-1)

def non_zero_mse(y_true, y_pred):
    t = y_true[y_true > 0]
    p = y_pred[y_true > 0]

    squared_error = (t - p) ** 2
    return tf.reduce_mean(squared_error, axis=-1)
