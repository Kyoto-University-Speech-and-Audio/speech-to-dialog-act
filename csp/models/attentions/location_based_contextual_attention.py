from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

class LocationBasedAttention(_BaseAttentionMechanism):
    def __init__(self, num_units,
                 memory, memory_sequence_length, context,
                 sharpening=False, smoothing=True,
                 use_location_based_attention=True,
                 location_conv_size=(10, 201),
                 scale=False):
        if not smoothing:
            probability_fn = lambda score, _: tf.nn.softmax(score)
        else:
            def sigmoid_prob(score, _):
                sigmoids = tf.sigmoid(score)
                return sigmoids / tf.reduce_sum(sigmoids, axis=-1,
                                                keep_dims=True)

            probability_fn = sigmoid_prob

        if not sharpening:
            memory_layer = layers_core.Dense(num_units, name="memory_layer", use_bias=False, dtype=tf.float32)
        # else:
        # memory_layer = layers_core.Dense()

        super(LocationBasedAttention, self).__init__(
            query_layer=layers_core.Dense(num_units, name="query_layer", use_bias=False, dtype=tf.float32),
            memory_layer=memory_layer,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=None,
            probability_fn=probability_fn,
            name="CustomAttention"
        )

        self.num_units = num_units
        self.location_conv_size = location_conv_size
        self.use_location_based_attention = use_location_based_attention
        self._scale = scale
        self._context = context

    def __call__(self, query, state):
        with tf.variable_scope(None, "custom_attention", [query]):
            processed_query = self.query_layer(query)
            processed_query = tf.expand_dims(processed_query, 1)  # W * s_{i-1}

            if self.use_location_based_attention:
                # expanded_alignments = tf.expand_dims(tf.expand_dims(state, axis=-1), axis=-1)
                # f = tf.layers.conv2d(expanded_alignments, self.num_units,
                #        self.location_conv_size, padding='same',
                #        use_bias=False, name='location_conv')
                # f = tf.squeeze(f, [2])
                # processed_location = tf.layers.dense(f, self.num_units,
                #        use_bias=False, name='location_layer')
                expanded_alignments = tf.expand_dims(state, axis=-1)
                f = tf.layers.conv1d(expanded_alignments,
                                     self.location_conv_size[1],
                                     [self.location_conv_size[0]],
                                     padding='same', use_bias=False, name='location_conv')
                processed_location = tf.layers.dense(f,
                                                     self.num_units,
                                                     use_bias=False, name='location_layer')  # U * f_{i, j}
            else:
                processed_location = tf.no_op()

            # b = tf.get_variable("attention_b", [self.num_units], dtype=tf.float32, initializer=tf.zeros_initializer)
            v = tf.get_variable("attention_v", [self.num_units], dtype=tf.float32)
            score = tf.reduce_sum(v * tf.tanh(processed_query +
                                              processed_location + self.keys), [2])

            alignments = self._probability_fn(score, state)

            next_state = alignments
            return alignments, next_state