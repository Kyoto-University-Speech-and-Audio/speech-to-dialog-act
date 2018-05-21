import tensorflow as tf

def sparse_tensor(densed_tensor, padding_value=0):
    # indices = tf.where(tf.not_equal(densed_tensor, tf.constant(-1, densed_tensor.dtype)))
    indices = tf.where(tf.not_equal(densed_tensor, tf.constant(padding_value, densed_tensor.dtype)))
    values = tf.gather_nd(densed_tensor, indices)
    shape = tf.shape(densed_tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)

def pad_tensor(tensor, new_size, value, axis=1):
    shape = tf.shape(tensor)
    return tf.concat([
        tensor,
        tf.fill(tf.concat([shape[:axis], [new_size - shape[axis]], shape[axis + 1:]], 0), value)
    ], axis)
