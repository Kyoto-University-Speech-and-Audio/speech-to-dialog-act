import tensorflow as tf

def sparse_tensor(densed_tensor, padding_value=0):
    # indices = tf.where(tf.not_equal(densed_tensor, tf.constant(-1, densed_tensor.dtype)))
    indices = tf.where(tf.not_equal(densed_tensor, tf.constant(padding_value, densed_tensor.dtype)))
    values = tf.gather_nd(densed_tensor, indices)
    shape = tf.shape(densed_tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)

def pad_tensor(tensor, new_size, value, axis=1):
    shape = tf.shape(tensor)
    return tf.cond(
        new_size > shape[axis],
        lambda: tf.concat([
            tensor,
            tf.fill(tf.concat([shape[:axis], [new_size - shape[axis]], shape[axis + 1:]], axis=0), value)
        ], axis),
        lambda: tensor)

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for (i, c1) in enumerate(s1):
        current_row = [i + 1]
        for (j, c2) in enumerate(s2):
            insertions = previous_row[j + 1] + 1  # j+1 instead of j since
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]