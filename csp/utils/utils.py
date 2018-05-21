import tensorflow as tf

def get_batched_input_class(FLAGS):
    BatchedInput = None

    if FLAGS.dataset == 'vivos':
        from ..input_data.vivos import BatchedInput
    elif FLAGS.dataset == 'vctk':
        from ..input_data.vctk import BatchedInput
    elif FLAGS.dataset == 'aps':
        from ..input_data.aps import BatchedInput

    return BatchedInput

def get_model_class(FLAGS):
    Model = None

    if FLAGS.model == 'ctc':
        from ..models.ctc import CTCModel as Model
    elif FLAGS.model == 'attention':
        from ..models.attention import AttentionModel as Model
    elif FLAGS.model == 'ctc-attention':
        from ..models.ctc_attention import CTCAttentionModel as Model

    return Model

def get_batched_dataset(dataset, batch_size, coef_count, num_buckets, mode):
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(([None, coef_count], []),
                           ([None], [])))

    def batching_func_infer(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=([None, coef_count], []))

    if num_buckets > 1:
        def key_func(src, tgt):
            bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(src[1] // bucket_width, tgt[1] // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def key_func_infer(src, src_len):
            bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = src_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        def reduce_func_infer(unused_key, windowed_data):
            return batching_func_infer(windowed_data)

        # self.batched_dataset = self.batched_dataset.apply(tf.contrib.data.batch_and_drop_remainder(hparams.batch_size))
        if mode == tf.estimator.ModeKeys.PREDICT:
            return dataset.apply(
                tf.contrib.data.group_by_window(
                    key_func=key_func_infer, reduce_func=reduce_func_infer, window_size=batch_size))
        else:
            return dataset.apply(
                tf.contrib.data.group_by_window(
                    key_func=key_func, reduce_func=reduce_func, window_size=batch_size))