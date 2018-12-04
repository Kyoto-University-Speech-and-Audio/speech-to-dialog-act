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

def pad_and_concat(t1, t2, axis=0, rank=1):
    new_size = tf.maximum(tf.shape(t1)[1], tf.shape(t2)[1])
    return tf.concat([
        tf.pad(t1, [[0, 0], [0, new_size - tf.shape(t1)[1]]] + [[0, 0]] * rank),
        tf.pad(t2, [[0, 0], [0, new_size - tf.shape(t2)[1]]] + [[0, 0]] * rank)
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


def levenshtein_no_sub(s,t):
    s = [None] + s
    t = [None] + t
    d = {}
    S = len(s)
    T = len(t)
    for i in range(S):
        d[i, 0] = i
    for j in range(T):
        d[0, j] = j
    for j in range(1, T):
        for i in range(1, S):
            if s[i] == t[j]: d[i, j] = d[i - 1, j - 1]
            else: d[i, j] = min(d[i - 1, j], d[i, j - 1]) + 1
    return d[S - 1, T - 1]


def calculate_ler(target, result, decode_fn, id=None, dist_fn=levenshtein):
    str_original = decode_fn(target)
    str_decoded = decode_fn(result)
    str_original = list(filter(lambda it: it != '<sp>', str_original))
    str_decoded = list(filter(lambda it: it != '<sp>', str_decoded))
    with open('log.log', 'a') as f:
        f.write("Original: %s\nDecoded: %s\n" % (str_original, str_decoded))
    if len(str_original) != 0:
        ler = dist_fn(str_original, str_decoded) / len(str_original)
        return min(1.0, ler), str_original, str_decoded
    else: return None, str_original, str_decoded


def evaluate(target, result, decode_fn, metrics="wer", id=None):
    sos_id = 27286
    eos_id = 27285
    eot_id = 27287

    if metrics == "wer":
        if False:
            target = [t for t in target if t < 27287]
            result = [r for r in result if r < 27287]
        return calculate_ler(target, result, decode_fn, id)
    #elif metrics == "ser":  # segment error rate
    #    target = [1 if t < 27285 else t for t in target]
    #    result = [1 if t < 27285 else t for t in result]
    #    return calculate_ler(target, result, decode_fn, id)
    elif metrics == "ser_incorp":  # segment error rate
        t = []
        for _t in target:
            if _t < 27285: t.append(1)
            elif _t == 27287: t[-1] = 2

        r = []
        id = 1
        for _r in result:
            if _r < 27285: r.append(1)
            elif _r == 27287: r[-1] = 2
        return calculate_ler(t, r, decode_fn, id, levenshtein_no_sub)
    elif metrics == "ser1":
        t_ids = [i for i, t in enumerate(target) if t == 2]
        r_ids = [i for i, r in enumerate(result) if r == 2]
        s = 0
        for t in t_ids: s += min([abs(r - t) for r in r_ids])
        for r in r_ids: s += min([abs(r - t) for t in t_ids])
        return s / 2 / len(target), decode_fn(target), decode_fn(result)
    elif metrics == "ser1_incorp":
        target = decode_fn(target)
        result = decode_fn(result)
        t_ids = []
        c = 1
        for i, t in enumerate(target):
            if t[:2] == "</":
                t_ids.append(i - c)
                c += 1
        r_ids = []
        c = 1
        for i, r in enumerate(result):
            if r[:2] == "</":
                r_ids.append(i - c)
                c += 1
        s = 0
        for t in t_ids: s += min([abs(r - t) for r in r_ids])
        for r in r_ids: s += min([abs(r - t) for t in t_ids])
        return s / 2 / len(target), target, result
    elif metrics == "ser":  # segment error rate
        target = [t for t in target if t == 1 or t == 2]
        result = [r for r in result if r == 1 or r == 2]
        return calculate_ler(target, result, decode_fn, id, levenshtein_no_sub)
    elif metrics == "scer_incorp": # segment count error rate
        target = [t for t in target if t == 27287]
        result = [t for t in result if t == 27287]
        return calculate_ler(target, result, decode_fn, id, levenshtein_no_sub)
    elif metrics == "scer":  # segment error rate
        target = [t for t in target if t == 2]
        result = [r for r in result if r == 2]
        return calculate_ler(target, result, decode_fn, id, levenshtein_no_sub)
    elif metrics == "wer_notag":
        target = decode_fn(target)
        target = list(filter(lambda t: t[:2] != '</', target))
        result = decode_fn(result)
        result = list(filter(lambda t: t[:2] != '</', result))
        if len(target) != 0:
            ler = levenshtein(target, result) / len(target)
            return min(1.0, ler), target, result
    elif metrics == "ter":  
        # tag error rate
        target = decode_fn(target)
        cur_tag = None
        t = []
        for i in reversed(range(len(target))):
            if target[i][:2] == '</':
                if target[i] != '</da>': # tag
                    cur_tag = target[i]
            else: 
                if cur_tag is not None: t.append(cur_tag)

        result = decode_fn(result)
        cur_tag = None
        r = []
        for i in reversed(range(len(result))):
            if result[i][:2] == '</':
                if result[i] != '</da>': # tag
                    cur_tag = result[i]
            else: 
                if cur_tag is not None: r.append(cur_tag)

        if len(t) != 0:
            ler = levenshtein(t, r) / len(t)
            return min(1.0, ler), t, r
        else: return None, t, r
    
    elif metrics == "ter_incl":  
        # tag error rate (for incorporating with ASR for _d in d)
        target = decode_fn(target, id)
        cur_tag = '</da>'
        t = []
        for i in reversed(range(len(target))):
            if target[i][:2] == '</': # tag
                cur_tag = target[i]
            t.append(cur_tag)

        result = decode_fn(result, id)
        #result = result[:len(target)]
        cur_tag = '</da>'
        r = []
        for i in reversed(range(len(result))):
            if result[i][:2] == '</': # tag
                cur_tag = result[i]
            r.append(cur_tag)

        if len(t) != 0:
            ler = levenshtein(t, r) / len(t)
            return min(1.0, ler), t, r
        else: return None, t, r

    elif metrics == "soer":
        # segmentation only error rate
        target = [t for t in target if t != 0]
        result = result[:len(target)]
        start = -1
        count = 0

        for i in range(len(target)):
            if target[i] == result[i] == 2:
                if start != None: count += i - start
                start = i
            elif target[i] == 2 or result[i] == 2: start = None
        return min(1 - count / len(target), 1.0), \
                decode_fn(target), decode_fn(result)

def joint_evaluate(hparams, target1, result1, target2, result2, decode_fn):
    if hparams.model == "da_attention_seg":
        sos_id = 27286
        eos_id = 27285
        eot_id = 27287
        i = 0
        t = []
        target1 = list(filter(lambda l: l != sos_id and l != eos_id, target1))
        #print(target2)
        for da in target2:
            while i < len(target1) and target1[i] != eot_id:
                i += 1
                t.append(da)
            i += 1

        i = 0
        r = []
        result1 = list(filter(lambda l: l != eos_id and l != sos_id, result1))
        #print(result2)
        for da in result2:
            while i < len(result1) and result1[i] != eot_id:
                i += 1
                r.append(da)
            i += 1
    
        return calculate_ler(t, r, decode_fn, 1)

