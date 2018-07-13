import tensorflow as tf
from .. import configs

def get_batched_input_class(hparams):
    BatchedInput = None

    if hparams.dataset == 'vivos':
        from ..datasets.vivos_feature import BatchedInput
    elif hparams.dataset == 'vctk':
        from ..datasets.vctk import BatchedInput
    elif hparams.dataset == 'aps':
        from ..datasets.aps import BatchedInput
    elif hparams.dataset == 'swbd':
        from ..datasets.swbd import BatchedInput
    elif hparams.dataset == 'aps-word':
        from ..datasets.aps import BatchedInput
    elif hparams.dataset == 'erato':
        from ..datasets.erato import BatchedInput
    elif hparams.dataset == 'erato_context':
        from ..datasets.erato_context import BatchedInput
    elif hparams.dataset == 'erato_prev_utt':
        from ..datasets.erato_prev_utt import BatchedInput
    return BatchedInput


def get_model_class(hparams):
    Model = None
    if hparams.model == 'ctc':
        from ..models.ctc import CTCModel as Model
    elif hparams.model == 'attention':
        from ..models.attention import AttentionModel
        return AttentionModel
    elif hparams.model == 'attention_monotonic_luong':
        from ..models.attention_monotonic import LuongMonotonicAttentionModel
        return LuongMonotonicAttentionModel
    elif hparams.model == 'attention_monotonic_bahdanau':
        from ..models.attention_monotonic import BahdanauMonotonicAttentionModel
        return BahdanauMonotonicAttentionModel
    elif hparams.model == 'attention_correction':
        from ..models.attention_correction import AttentionModel
        return AttentionModel
    elif hparams.model == 'attention_context':
        from ..models.attention_context import AttentionModel
        return AttentionModel
    elif hparams.model == 'attention_prev_utt':
        from ..models.attention_prev_utt_lstm import AttentionModel
        return AttentionModel
    elif hparams.model == 'attention_prev_utt_is':
        from ..models.attention_prev_utt_lstm_is import AttentionModel
        return AttentionModel
    elif hparams.model == 'attention_keep_encoder':
        from ..models.attention_keep_encoder import AttentionModel
        return AttentionModel
    elif hparams.model == 'ctc-attention':
        from ..models.ctc_attention import CTCAttentionModel as Model
    return Model


def get_optimizer(hparams, learning_rate):
    if hparams.optimizer == "sgd":
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif hparams.optimizer == "adam":
        return tf.train.AdamOptimizer(learning_rate)
    elif hparams.optimizer == "momentum":
        return tf.train.MomentumOptimizer(learning_rate, 0.9)


def get_batched_dataset(dataset, batch_size, coef_count, mode, padding_values=0):
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(([], [None, coef_count], []),
                       ([None], [])),
        padding_values=(('', 0.0, 0), (padding_values, 0))
    )
    #if mode == tf.estimator.ModeKeys.PREDICT:
    #    dataset = dataset.filter(lambda x: tf.equal(tf.shape(x[0]), batch_size))
    #else:
    #    dataset = dataset.filter(lambda x, y: tf.equal(tf.shape(x[0])[0], batch_size))
    return dataset


def get_batched_dataset_bucket(dataset, batch_size, coef_count, num_buckets, mode, padding_values=0):
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(([], [None, coef_count], []),
                           ([None], [])),
            padding_values=(('', 0.0, 0), (padding_values, 0))
        )

    def batching_func_infer(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=([], [None, coef_count], []))

    if num_buckets > 1:
        def key_func(src, tgt):
            bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(src[2] // bucket_width, tgt[1] // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def key_func_infer(fn, src, src_len):
            bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = src_len // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        def reduce_func_infer(unused_key, windowed_data):
            return batching_func_infer(windowed_data)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return dataset.apply(
                tf.contrib.data.group_by_window(
                    key_func=key_func_infer, reduce_func=reduce_func_infer, window_size=batch_size))
        else:
            return dataset.apply(
                tf.contrib.data.group_by_window(
                    key_func=key_func, reduce_func=reduce_func, window_size=batch_size))


def argval(name, flags):
    if hasattr(flags, name):
        return getattr(flags, name)
    else:
        return None


def create_hparams(flags):
    def _argval(name):
        return argval(name, flags)

    hparams = tf.contrib.training.HParams(
        model=_argval('model'),
        dataset=_argval('dataset'),
        name=_argval('name'),
        input_unit=_argval('input_unit'),
        verbose=_argval('verbose') or False,

        batch_size=_argval('batch_size') or 32,
        eval_batch_size=_argval('batch_size') or 32,
        num_buckets=5,

        sample_rate=16000,
        window_size_ms=30.0,
        window_stride_ms=10.0,

        epoch_step=0,

        summaries_dir=None,
        out_dir=None,
        beam_width=4,
        sampling_temperature=0.0,
        num_units=320,
        num_encoder_layers=3,
        num_decoder_layers=1,
        num_classes=0,
        num_features=120,

        colocate_gradients_with_ops=True,

        learning_rate=_argval("learning_rate") or 1e-3,
        optimizer="adam",
        max_gradient_norm=5.0,

        # Data
        vocab_file=None,
        train_data=None,
        eval_data=None,
        encoding="euc-jp",

        # Attention
        sos_index=1,
        eos_index=2,
        encoder_type='lstm',
        decoder_num_units=320,
        encoder_num_units=320,
        attention_layer_size=128,
        attention_energy_scale=False,
        attention_num_units=128,
        output_attention=False,

        # Infer
        input_path=_argval("input_path") or configs.DEFAULT_INFER_INPUT_PATH,
        hcopy_path=None,
        hcopy_config=None,
        length_penalty_weight=0.0,

        load=_argval('load'),
        shuffle=_argval('shuffle')
    )

    if flags.config is not None:
        json = open('model_configs/%s.json' % flags.config).read()
        hparams.parse_json(json)

    hparams.summaries_dir = "log/" + hparams.name
    hparams.out_dir = "saved_models/" + hparams.name

    tf.logging.info(hparams)

    return hparams


def clear_log(path=None):
    f = open(path or configs.DEFAULT_LOG_PATH, 'w')
    f.close()


def write_log(text, path=None):
    f = open(path or configs.DEFAULT_LOG_PATH, 'a')
    f.write('\n'.join(text) + '\n')
    f.close()
