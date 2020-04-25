from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import numpy as np

from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

from .attention import AttentionModel as BaseAttentionModel
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _BaseAttentionMechanism, _bahdanau_score, monotonic_attention

"""Example soft monotonic alignment decoder implementation.
This file contains an example TensorFlow implementation of the approach
described in ``Online and Linear-Time Attention by Enforcing Monotonic
Alignments''.  The function monotonic_attention covers the algorithms in the
paper and should be general-purpose.  monotonic_alignment_decoder can be used
directly in place of tf.nn.seq2seq.attention_decoder.  This implementation
attempts to deviate as little as possible from tf.nn.seq2seq.attention_decoder,
in order to facilitate comparison between the two decoders.
"""
import tensorflow as tf


class _BaseMonotonicAttentionMechanism(_BaseAttentionMechanism):
    """Base attention mechanism for monotonic attention.
    Simply overrides the initial_alignments function to provide a dirac
    distribution, which is needed in order for the monotonic attention
    distributions to have the correct behavior.
    """

    def initial_alignments(self, batch_size, dtype):
        """Creates the initial alignment values for the monotonic attentions.
        Initializes to dirac distributions, i.e. [1, 0, 0, ...memory length..., 0]
        for all entries in the batch.
        Args:
        batch_size: `int32` scalar, the batch_size.
        dtype: The `dtype`.
        Returns:
        A `dtype` tensor shaped `[batch_size, alignments_size]`
        (`alignments_size` is the values' `max_time`).
        """
        max_time = self._alignments_size
        return array_ops.one_hot(
            array_ops.zeros((batch_size,), dtype=dtypes.int32), max_time,
            dtype=dtype)


def _monotonic_probability_fn(score, previous_alignments, sigmoid_noise, mode,
                              seed=None):
    """Attention probability function for monotonic attention.
    Takes in unnormalized attention scores, adds pre-sigmoid noise to encourage
    the model to make discrete attention decisions, passes them through a sigmoid
    to obtain "choosing" probabilities, and then calls monotonic_attention to
    obtain the attention distribution.  For more information, see
    Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  https://arxiv.org/abs/1704.00784
    Args:
        score: Unnormalized attention scores, shape `[batch_size, alignments_size]`
        previous_alignments: Previous attention distribution, shape
        `[batch_size, alignments_size]`
        sigmoid_noise: Standard deviation of pre-sigmoid noise.  Setting this larger
        than 0 will encourage the model to produce large attention scores,
        effectively making the choosing probabilities discrete and the resulting
        attention distribution one-hot.  It should be set to 0 at test-time, and
        when hard attention is not desired.
        mode: How to compute the attention distribution.  Must be one of
        'recursive', 'parallel', or 'hard'.  See the docstring for
        `tf.contrib.seq2seq.monotonic_attention` for more information.
        seed: (optional) Random seed for pre-sigmoid noise.
    Returns:
        A `[batch_size, alignments_size]`-shape tensor corresponding to the
        resulting attention distribution.
    """
    # Optionally add pre-sigmoid noise to the scores
    if sigmoid_noise > 0:
        noise = random_ops.random_normal(array_ops.shape(score), dtype=score.dtype,
                                        seed=seed)
        score += sigmoid_noise*noise
    # Compute "choosing" probabilities from the attention scores
    if mode == "hard":
        # When mode is hard, use a hard sigmoid
        p_choose_i = math_ops.cast(score > 0, score.dtype)
    else:
        p_choose_i = math_ops.sigmoid(score)
    # Convert from choosing probabilities to attention distribution
    return monotonic_attention(p_choose_i, previous_alignments, mode)


class BahdanauMonotonicAttention(_BaseMonotonicAttentionMechanism):
    """Monotonic attention mechanism with Bahadanau-style energy function.
    This type of attention enforces a monotonic constraint on the attention
    distributions; that is once the model attends to a given point in the memory
    it can't attend to any prior points at subsequence output timesteps.  It
    achieves this by using the _monotonic_probability_fn instead of softmax to
    construct its attention distributions.  Since the attention scores are passed
    through a sigmoid, a learnable scalar bias parameter is applied after the
    score function and before the sigmoid.  Otherwise, it is equivalent to
    BahdanauAttention.  This approach is proposed in
    Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  https://arxiv.org/abs/1704.00784
    """

    def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               score_mask_value=None,
               sigmoid_noise=0.,
               sigmoid_noise_seed=None,
               score_bias_init=0.,
               mode="parallel",
               dtype=None,
               name="BahdanauMonotonicAttention"):
        """Construct the Attention mechanism.
        Args:
        num_units: The depth of the query mechanism.
        memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
        memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
        normalize: Python boolean.  Whether to normalize the energy term.
        score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
        sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the docstring
            for `_monotonic_probability_fn` for more information.
        sigmoid_noise_seed: (optional) Random seed for pre-sigmoid noise.
        score_bias_init: Initial value for score bias scalar.  It's recommended to
            initialize this to a negative value when the length of the memory is
            large.
        mode: How to compute the attention distribution.  Must be one of
            'recursive', 'parallel', or 'hard'.  See the docstring for
            `tf.contrib.seq2seq.monotonic_attention` for more information.
        dtype: The data type for the query and memory layers of the attention
            mechanism.
        name: Name to use when creating ops.
        """
        # Set up the monotonic probability fn with supplied parameters
        if dtype is None:
            dtype = dtypes.float32
        wrapped_probability_fn = functools.partial(
            _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
            seed=sigmoid_noise_seed)
        super(BahdanauMonotonicAttention, self).__init__(
            query_layer=layers_core.Dense(
                num_units, name="query_layer", use_bias=False, dtype=dtype),
            memory_layer=layers_core.Dense(
                num_units, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name
        self._score_bias_init = score_bias_init

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
        query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
        state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
        alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with variable_scope.variable_scope(None, "bahdanau_monotonic_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _bahdanau_score(processed_query, self._keys, self._normalize)
            score_bias = variable_scope.get_variable(
                "attention_score_bias", dtype=processed_query.dtype,
                initializer=self._score_bias_init)
            score += score_bias
        
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state


class LuongMonotonicAttentionModel(BaseAttentionModel):
    def __init__(self):
        super().__init__(attention_fn=self._attention_fn)

    def _attention_fn(self, input_seq_len, encoder_outputs):
        return tf.contrib.seq2seq.LuongMonotonicAttention(
            self.hparams.attention_num_units,
            encoder_outputs,
            memory_sequence_length=input_seq_len,
            scale=self.hparams.attention_energy_scale,
            score_bias_init=-4.0,
            sigmoid_noise=1,
            mode='hard'
        )


class BahdanauMonotonicAttentionModel(BaseAttentionModel):
    def __init__(self):
        super().__init__(
            attention_fn=self._monotonic_attention_fn,
            train_decode_fn=self._monotonic_train_decode_fn)

    def _monotonic_attention_fn(self, input_seq_len, encoder_outputs):
        return tf.contrib.seq2seq.BahdanauMonotonicAttention(
            self.hparams.attention_num_units,
            encoder_outputs,
            memory_sequence_length=input_seq_len,
            score_bias_init=-3.0,
            sigmoid_noise=1,
            mode='hard'
        )

    def _monotonic_train_decode_fn(
            self, decoder_inputs, target_seq_len,
            encoder_outputs, encoder_final_state, 
            decoder_cell,
            scope, context=None):
        attention_cell = self._get_attention_cell(
            decoder_cell,
            encoder_outputs,
            self.input_seq_len
        )

        helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, target_seq_len)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            attention_cell, 
            helper, 
            self._get_decoder_initial_state(attention_cell, encoder_final_state))

        outputs, final_state, final_output_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            swap_memory=True,
            scope=scope)

        return attention_cell, outputs, final_state, final_output_lengths