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

def pad_and_concat(t1, t2, axis=0):
    new_size = tf.maximum(tf.shape(t1)[1], tf.shape(t2)[1])
    return tf.concat([
        tf.pad(t1, [[0, 0], [0, new_size - tf.shape(t1)[1]]] + [[0, 0]] * 1),
        tf.pad(t2, [[0, 0], [0, new_size - tf.shape(t2)[1]]] + [[0, 0]] * 1)
    ], axis)

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

def calculate_ler(target, result, decode_fn, id=None):
    str_original = decode_fn(target)
    str_decoded = decode_fn(result)
    str_original = list(filter(lambda it: it != '<sp>', str_original))
    str_decoded = list(filter(lambda it: it != '<sp>', str_decoded))
    with open('log.log', 'a') as f:
        f.write("Original: %s\nDecoded: %s\n" % (str_original, str_decoded))
    if len(str_original) != 0:
        # ler = ops_utils.levenshtein(''.join(str_original), ''.join(str_decoded)) / len(''.join(str_original))
        # ler = ops_utils.levenshtein(target_labels[i], decoded[i]) / len(target_labels[i])
        # ler = levenshtein(' '.join(str_original), ' '.join(str_decoded)) / len(' '.join(str_original))
        ler = levenshtein(str_original, str_decoded) / len(str_original)
        return min(1.0, ler), str_original, str_decoded
    else: return None, str_original, str_decoded

def evaluate(target, result, decode_fn, metrics="wer", id=None):
    if metrics == "wer":
        return calculate_ler(target, result, decode_fn, id)
    elif metrics == "ser":  # segment error rate
        target = [1 if t < 27285 else t for t in target]
        result = [1 if t < 27285 else t for t in result]
        return calculate_ler(target, result, decode_fn, id)
    elif metrics == "scer": # segment count error rate
        target = [t for t in target if t == 27287]
        result = [t for t in result if t == 27287]
        return calculate_ler(target, result, decode_fn, id)