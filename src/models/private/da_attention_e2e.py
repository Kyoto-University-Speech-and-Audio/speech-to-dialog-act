from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq import BasicDecoderOutput, BasicDecoder as BasicDecoderBase
from tensorflow.contrib.seq2seq import TrainingHelper as TrainingHelperBase, GreedyEmbeddingHelper as GreedyEmbeddingHelperBase

from ..attention import AttentionModel
from .da import Model as DAModel


class BasicDecoder(BasicDecoderBase):
    def __init__(self, cell, helper, initial_state, output_layer=None):
        """Initialize BasicDecoder.
        Args:
        cell: An `RNNCell` instance.
        helper: A `Helper` instance.
        initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
            The initial state of the RNNCell.
        output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`. Optional layer to apply to the RNN output prior
            to storing the result or sampling.
        Raises:
        TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
        """
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = None
        self.__output_layer = output_layer

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = nest.map_structure(
                lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                size)
            layer_output_shape = self._output_layer.compute_output_shape(
                output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicDecoderOutput(
            rnn_output=self._rnn_output_size(),
            sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = nest.flatten(self._initial_state)[0].dtype
        return BasicDecoderOutput(
            nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self._helper.sample_ids_dtype)

    def initialize(self, name=None):
        """Initialize the decoder.
        Args:
        name: Name scope for any created operations.
        Returns:
        `(finished, first_inputs, initial_state)`.
        """
        return self._helper.initialize() + (self._initial_state, )

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
        Args:
        time: scalar `int32` tensor.
        inputs: A (structure of) input tensors.
        state: A (structure of) state tensors and TensorArrays.
        name: Name scope for any created operations.
        Returns:
        `(outputs, next_state, next_inputs, finished)`.
        """
        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            #cell_outputs = 
            cell_dist_outputs = self.__output_layer(cell_outputs) if self.__output_layer else cell_outputs
            sample_ids = self._helper.sample(
                time=time, outputs=cell_dist_outputs, state=cell_state)
            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                dist_outputs=cell_dist_outputs,
                state=cell_state,
                sample_ids=sample_ids)
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)

def _unstack_ta(inp):
    return tf.TensorArray(
            dtype=inp.dtype, size=tf.shape(inp)[0],
            element_shape=inp.get_shape()[1:]).unstack(inp)

class TrainingHelper(TrainingHelperBase):
    def __init__(self, inputs, input_size, sequence_length, sos_emb, time_major=False,
            name=None, output_layer=None):
        """Initializer.
        Args:
        inputs: not used.
        sequence_length: An int32 vector tensor.
        time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
        name: Name scope for any created operations.
        """

        with ops.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
            self._inputs = tf.convert_to_tensor(inputs, name="inputs")
            self._zero_inputs = tf.zeros_like(inputs[:, 0, :])

            self._sequence_length = tf.convert_to_tensor(
                sequence_length, name="sequence_length")
            self._batch_size = tf.size(sequence_length)
            self._input_size = tf.convert_to_tensor(input_size)
            self._start_inputs = tf.tile(tf.expand_dims(sos_emb, 0), [self._batch_size, 1])
            self._sos_emb = sos_emb
            self._output_layer = output_layer

    def initialize(self, name=None):
        with ops.name_scope(name, "TrainingHelperInitialize"):
            finished = tf.equal(0, self._sequence_length)
            all_finished = tf.reduce_all(finished)
            next_inputs_1 = self._start_inputs
            next_inputs_2 = tf.cond(
                        all_finished,
                        lambda: self._zero_inputs,
                        lambda: self._inputs[:, 0, :]
                    )
            #next_inputs = tf.concat([next_inputs_1, next_inputs_2], -1)
            next_inputs = next_inputs_1 + next_inputs_2
            return (finished, next_inputs)

    def next_inputs(self, time, outputs, dist_outputs, state, name=None, **unused_kwargs):
        """next_inputs_fn for TrainingHelper."""
        with ops.name_scope(name, "TrainingHelperNextInputs",
                            [time, outputs, state]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = tf.reduce_all(finished)
            outputs = self._output_layer(outputs)
            def read_from_ta(inp): return inp.read(next_time)

            next_inputs_1 = tf.cond(
                        all_finished, 
                        lambda: self._start_inputs,
                        lambda: outputs)
            next_inputs_2 = tf.cond(
                        all_finished,
                        lambda: self._zero_inputs,
                        lambda: self._inputs[:, next_time, :])

            next_inputs = next_inputs_1 + next_inputs_2
            #next_inputs = tf.concat([next_inputs_1, next_inputs_2], -1)
                                        
            return (finished, next_inputs, state)


class GreedyEmbeddingHelper(GreedyEmbeddingHelperBase):
    def __init__(self, embedding, start_tokens, batch_size, input_size, end_token, sos_emb,
            output_layer=None):
        self._embedding_fn = embedding
        self._batch_size = batch_size
        self._start_tokens = ops.convert_to_tensor(start_tokens,
                dtype=tf.int32, name="start_tokens")
        self._start_inputs_regular = self._embedding_fn(self._start_tokens)

        self._start_inputs = tf.tile(tf.expand_dims(sos_emb, 0), [self._batch_size, 1])

        self._end_token = end_token
        self._sos_emb = sos_emb
        self._output_layer = output_layer

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs (for sampling) are logits, use argmax to get the most probable id
        sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        return sample_ids

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        next_inputs = self._start_inputs + self._sos_emb
        return (finished, self._start_inputs)

    def next_inputs(self, time, outputs, dist_outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time  # unused by next_inputs_fn
        finished = tf.equal(sample_ids, self._end_token)
        all_finished = tf.reduce_all(finished)
        outputs = self._output_layer(outputs)
        
        next_inputs_1 = tf.cond(
                    all_finished, 
                    lambda: self._start_inputs,
                    lambda: outputs)
        next_inputs_2 = tf.cond(
                    all_finished,
                    lambda: self._start_inputs_regular,
                    lambda: self._embedding_fn(sample_ids))

        next_inputs = next_inputs_1 + next_inputs_2

        return (finished, next_inputs, state)


class AttentionE2EModel(AttentionModel):
    def __init__(self):
        super().__init__(
            train_decode_fn=self._train_decode,
            eval_decode_fn=self._eval_decode,
            force_alignment_history=False
        )

    def _build_graph(self):
        self._output_transform_layer = tf.layers.Dense(self.hparams.decoder_num_units)
        return super()._build_graph()
    
    def _train_decode(
            self, decoder_inputs, target_seq_len,
            encoder_outputs, encoder_final_state, 
            decoder_cell,
            scope, context=None):
        self._sos_emb = tf.get_variable("sos_emb", [self.hparams.decoder_num_units], dtype=tf.float32)
        attention_cell = self._get_attention_cell(
            decoder_cell,
            encoder_outputs,
            self.input_seq_len
        )

        one_hot_targets = tf.one_hot(decoder_inputs, depth=self.hparams.vocab_size)
        helper = TrainingHelper(
                self.decoder_emb_layer(one_hot_targets), 
                self.hparams.decoder_num_units, target_seq_len,
                self._sos_emb, 
                output_layer=self._output_transform_layer)
        decoder = BasicDecoder(
            attention_cell, 
            helper, 
            self._get_decoder_initial_state(attention_cell, encoder_final_state)
        )

        outputs, final_state, final_output_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            swap_memory=True,
            scope=scope)

        return attention_cell, outputs, final_state, final_output_lengths

    def _eval_decode(
            self, 
            encoder_outputs, encoder_final_state,
            decoder_cell, scope, context=None):
        self._sos_emb = tf.get_variable("sos_emb", [self.hparams.decoder_num_units], dtype=tf.float32)
        if self.hparams.beam_width > 0:
            encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=self.hparams.beam_width)
            input_seq_len = tf.contrib.seq2seq.tile_batch(
                self.input_seq_len, multiplier=self.hparams.beam_width)
            batch_size = self.batch_size * self.hparams.beam_width
        else:
            input_seq_len = self.input_seq_len
            batch_size = self.batch_size

        attention_cell = self._get_attention_cell(decoder_cell, encoder_outputs, input_seq_len)
        initial_state = self._get_decoder_initial_state(attention_cell, encoder_final_state, batch_size)

        def embed_fn(ids):
            return self.decoder_emb_layer(tf.one_hot(ids, depth=self.hparams.vocab_size))

        helper = GreedyEmbeddingHelper(
            embed_fn,
            start_tokens=tf.fill([self.batch_size], self.hparams.sos_index),
            batch_size=self.batch_size,
            input_size=self.hparams.decoder_num_units,
            end_token=self.hparams.eos_index,
            sos_emb=self._sos_emb,
            output_layer=self._output_transform_layer
        )

        decoder = BasicDecoder(
            attention_cell,
            helper,
            initial_state=initial_state,
            output_layer=self.output_layer,
        )

        outputs, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            swap_memory=True,
            maximum_iterations=50,
            scope=scope)

        return attention_cell, outputs, final_context_state, final_sequence_lengths

    @classmethod
    def load(cls, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        var_list = {}
        for var in saver_variables:
            if var.op.name[:7] == "encoder":
                var_list[var.op.name] = var
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)

    @classmethod
    def ignore_save_variables(cls):
        ls = []
        for var in tf.global_variables():
            if var.op.name[:7] == "encoder" and (var.op.name[-4:] == "Adam" or
                    var.op.name[-6:] == "Adam_1"):
                ls.append(var.op.name)
        return ls

    def get_extra_ops(self):
        return [self.target_labels, tf.argmax(self.logits, -1)]

class Model(AttentionE2EModel, DAModel):
    default_params = {
        **AttentionE2EModel.default_params,
        **DAModel.default_params,
        'joint_training': True
    }
    def __init__(self):
        AttentionE2EModel.__init__(self)

    def __call__(self, hparams, mode, iterator, **kwargs):
        super(AttentionE2EModel, self).__call__(hparams, mode, iterator, **kwargs)
        return self
    
    def _assign_input(self):
        self.dlg_ids, (self.inputs, self.input_seq_len), (self.targets, self.target_seq_len), self.da_labels = self.iterator.get_next()
    
    def get_ground_truth_label_placeholder(self): return [self.targets, self.da_labels]

    def get_predicted_label_placeholder(self): return [self.sample_id, self.predicted_da_labels]

    def get_ground_truth_label_len_placeholder(self): return [self.target_seq_len, tf.constant(1)]

    def get_predicted_label_len_placeholder(self): return [self.final_sequence_lengths, tf.constant(1)]

    def get_decode_fns(self):
        return [
            lambda d: self._batched_input.decode(d),
            lambda d: self._batched_input.decode_da(d)
        ]

    def get_da_inputs(self, beam_id=None):
        da_inputs = self.decoder_outputs
        if self.hparams.embedding_size != self.hparams.decoder_num_units:
            da_inputs = tf.layers.dense(da_inputs, self.hparams.embedding_size)
        da_input_len = self.final_sequence_lengths
        return da_inputs, da_input_len

    def _build_graph(self):
        with tf.variable_scope("asr"):
            loss_asr = AttentionE2EModel._build_graph(self)
        
        with tf.variable_scope("da_recog"):
            da_inputs, da_input_len = self.get_da_inputs()
            
            history_targets, history_target_seq_len, history_seq_len = self._build_history(
                da_inputs,
                da_input_len,
                rank=1,
                dtype=tf.float32
            )

            history_inputs = self._build_word_encoder(
                history_targets,
                history_target_seq_len,
            )
        
            encoded_history = self._build_utt_encoder(history_inputs, history_seq_len)

            loss_da, self.predicted_da_labels = self._get_loss(encoded_history)
            with tf.control_dependencies([loss_da]):
                self.update_prev_inputs = self._build_update_prev_inputs(da_inputs, da_input_len)
        
        if loss_asr == 0.0:
            loss = loss_da
        else:
            loss = self.hparams.da_attention_lambda * loss_asr + (1 - self.hparams.da_attention_lambda) * loss_da
        return loss

    @classmethod
    def load(cls, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        var_list = {}

        for var in saver_variables:
            if var.op.name[:4] == "asr/":
                var_list[var.op.name[4:]] = var
                print(var.op.name[4:])
        #for var in saver_variables:
        #    if not (var.op.name[:4] == "asr/" and (var.op.name[-4:] == "Adam" or var.op.name[-6:] == "Adam_1")):
        #        print(var.op.name)
        #        var_list[var.op.name] = var
        print(var_list)
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)
    
    def trainable_variables(self):
        trainable_vars = tf.trainable_variables()
        if self.hparams.joint_training:
            return list(filter(lambda var: var.op.name[:11] != "asr/encoder", trainable_vars))
            return trainable_vars
        else:    
            return list(filter(lambda var: var.op.name[:8] == "da_recog",
            trainable_vars))

    def get_extra_ops(self):
        return [self.da_logits]
