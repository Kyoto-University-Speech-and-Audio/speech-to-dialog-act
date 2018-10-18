from .attention import AttentionModel
from .da import Model as DAModel

import tensorflow as tf
import numpy as np

from ..utils import ops_utils, model_utils

class Model(AttentionModel, DAModel):
    def __init__(self):
        AttentionModel.__init__(self, force_alignment_history=True)

    def __call__(self, hparams, mode, iterator, **kwargs):
        super(AttentionModel, self).__call__(hparams, mode, iterator, **kwargs)
        return self
    
    def _assign_input(self):
        self.dlg_ids, (self.inputs, self.input_seq_len), (self.targets, self.target_seq_len), (self.da_labels, self.da_label_len) = self.iterator.get_next()
    
    def get_ground_truth_label_placeholder(self): 
        return [self.targets, self.da_labels]

    def get_predicted_label_placeholder(self): 
        return [self.sample_id, self.predicted_da_labels]

    def get_da_inputs(self):
        if self.hparams.da_input == "decoder_output":
            da_inputs = self.decoder_outputs
            if self.hparams.embedding_size != self.hparams.decoder_num_units:
                da_inputs = tf.layers.dense(da_inputs, self.hparams.embedding_size)
            da_input_len = self.final_sequence_lengths

        return da_inputs, da_input_len

    def _build_history(self, inputs, input_seq_len, target_ids, rank=0, dtype=tf.int32):
        """
        Args:
            inputs: size [batch_size, target_max_len, ...]
            input_seq_len: size [batch_size]
            target_ids: [batch_size, target_max_len]
        
        Returns:
            history_targets: size [batch_size, history_size + 1, target_max_len, ...]
            history_target_seq_len: size [batch_size, history_size + 1]
            da_masks: [batch_size, history_len + 1, target_max_len]
            history_seq_len: size [batch_size]
        """
        self.prev_targets = tf.get_variable("prev_targets", [0, 0, self.hparams.embedding_size], dtype=dtype, trainable=False)
        self.prev_target_seq_len = tf.get_variable("prev_target_seq_len", [0], tf.int32, trainable=False)
        self.prev_dlg_ids = tf.get_variable("prev_dlg_ids", [0], dtype=tf.string, 
                initializer=tf.constant_initializer('', tf.string),
                trainable=False)
        self.prev_da_masks = tf.get_variable("prev_da_masks", [0, 0], tf.bool, trainable=False)
        #self.prev_speakers = tf.placeholder(tf.bool, name="prev_speakers")

        # filter input from previous batch with same dialog id
        same_id_mask = tf.equal(self.prev_dlg_ids, self.dlg_ids[0])
        prev_same_dlg_targets = tf.boolean_mask(self.prev_targets, same_id_mask)
        prev_same_dlg_target_seq_len = tf.boolean_mask(self.prev_target_seq_len, same_id_mask)
        prev_da_masks = tf.boolean_mask(self.prev_da_masks, same_id_mask)

        # concatenate previous batch with current batch
        full_targets = ops_utils.pad_and_concat(prev_same_dlg_targets, inputs, 0) # [num_utts, target_max_len, feature_size]
        full_target_seq_len = tf.concat([prev_same_dlg_target_seq_len, input_seq_len], 0) # [num_utts]
        da_mask = tf.equal(target_ids, self.hparams.da_seg_index)
        full_da_masks = tf.concat([prev_da_masks, da_mask], 0)

        # build history data
        def pad_history(t, i, rank):
            ret = t[tf.maximum(i - self.hparams.num_utt_history, 0):i + 1]
            return tf.pad(
                    ret, 
                    [[0, tf.maximum(0, self.hparams.num_utt_history - i)]] + [[0, 0]] * (rank - 1)
            )

        start_id = tf.shape(prev_same_dlg_targets)[0]
        end_id = tf.shape(full_targets)[0]
        history_targets = tf.map_fn(
                lambda i: pad_history(full_targets, i, 3), 
                tf.range(start_id, end_id),
                dtype=dtype
        ) # [batch_size, history_len + 1, max_utt_len, feature_size]
        history_target_seq_len = tf.map_fn(
                lambda i: pad_history(full_target_seq_len, i, 1),
                tf.range(start_id, end_id)
        ) # [batch_size, history_len + 1]
        history_da_masks = tf.map_fn(
                lambda i: pad_history(full_da_masks, i, 2),
                tf.range(start_id, end_id)
        ) # [batch_size, history_len + 1, max_utt_len]
        history_seq_len = tf.map_fn(
                lambda i: tf.minimum(i + 1, self.hparams.num_utt_history + 1),
                tf.range(start_id, end_id)
        )
        return history_targets, history_target_seq_len, history_da_masks, history_seq_len

    def _build_word_encoder(
            self, 
            history_targets,
            history_target_seq_len,
            da_masks,
            embedding_fn=None):
        """
        Args:
            history_targets: size [batch_size, history_len + 1, target_max_len]
            history_target_seq_len: size [batch_size, history_len + 1]
            da_masks: size [batch_size, history_len + 1, target_max_len]

        Returns:
            history_inputs: size [batch_size, history_len + 1, da_seg_len, num_hidden_units] 
                tensor of encoded utterances per batch
            last_da_seg_len
        """
        shape1 = tf.shape(history_targets)[0] # batch_size
        shape2 = tf.shape(history_targets)[1] # history_len + 1
        shape3 = tf.shape(history_targets)[2] # target_max_len

        # flatten history to feed into encoder at word-level
        utt_inputs = tf.reshape(history_targets, [shape1 * shape2, shape3, self.hparams.embedding_size])
        utt_input_seq_len = tf.reshape(history_target_seq_len, [shape1 * shape2])

        if embedding_fn is not None: utt_inputs = embedding_fn(utt_inputs)
        sent_encoder_outputs, _ = self._build_da_word_encoder(utt_inputs, utt_input_seq_len)
        # sent_encoder_outputs_reshaped: [batch_size, history_len + 1, target_max_len, encoder_num_units]

        # history_inputs: [batch_size, history_len + 1, da_seg_len, decoder_size]
        last_da_seg_len = tf.count_nonzero(da_masks[:, -1, :], axis=-1)
        tf.assert_equal(tf.cast(last_da_seg_len, tf.int32), self.da_label_len)
        da_masks = tf.reshape(da_masks, [shape1, -1]) # [batch_size, history_word_len]
        sent_encoder_outputs_reshaped = tf.reshape(sent_encoder_outputs, [shape1, shape2 * shape3, sent_encoder_outputs.get_shape()[-1]]) # [batch_size, history_word_len, encoder_num_units]

        # history_inputs = tf.boolean_mask(sent_encoder_outputs_reshaped, da_masks, axis=1) # [batch_size, history_da_seg_len, decoder_size]
        history_input_len = tf.count_nonzero(da_masks, axis=-1)
        history_input_max_len = tf.reduce_max(history_input_len)
        history_inputs = tf.map_fn(
            lambda x: tf.pad(
                tf.boolean_mask(x[0], x[1]), 
                tf.stack([tf.stack([0, history_input_max_len - tf.count_nonzero(x[1])]), tf.constant([0, 0], tf.int64)])),
            (sent_encoder_outputs_reshaped, da_masks), 
            dtype=(tf.float32)
        )
        history_inputs = tf.reshape(history_inputs, [shape1, -1, sent_encoder_outputs.get_shape()[-1]])
        return history_inputs, history_input_len, last_da_seg_len

    def _build_utt_encoder(self, history_inputs, history_input_seq_len, last_da_seg_len):
        """
        Args:
            history_inputs: [batch_size, history_da_seg_len, num_hidden_units]
            history_input_seq_len: size [batch_size]
            last_da_seg_len < history_da_seg_len

        Returns:
            size [batch_size, last_da_seg_len, da_encoder_size] 
        """
        # encoder at utterance-level
        history_encoder_outputs, _ = tf.nn.dynamic_rnn(
                model_utils.single_cell("lstm",
                    self.hparams.utt_encoder_num_units, self.mode,
                    dropout=self.hparams.dropout), 
                history_inputs, 
                sequence_length=history_input_seq_len,
                dtype=tf.float32)
        return history_encoder_outputs
        #print(last_da_seg_len)
        #return tf.map_fn(lambda x: x[0][-x[1]:, :],
        #    (history_encoder_outputs, last_da_seg_len), tf.float32)
        
    def _build_graph(self):
        with tf.variable_scope("asr"):
            loss_asr = AttentionModel._build_graph(self)
        
        with tf.variable_scope("da_recog"):
            da_inputs, da_input_len = self.get_da_inputs()
            
            history_targets, history_target_seq_len, da_masks, history_seq_len = self._build_history(
                da_inputs,
                da_input_len,
                self.target_labels,
                rank=1,
                dtype=tf.float32
            )

            history_inputs, history_input_len, last_da_seg_len = self._build_word_encoder(
                history_targets,
                history_target_seq_len,
                da_masks
            )
        
            encoded_history = self._build_utt_encoder(history_inputs, history_input_len, last_da_seg_len)
            print(encoded_history)

            loss_da, self.predicted_da_labels = self._compute_da_loss(encoded_history)
            with tf.control_dependencies([loss_da]):
                self.update_prev_inputs = self._build_update_prev_inputs(da_inputs)
        
        loss = self.hparams.da_attention_lambda * loss_asr + (1 - self.hparams.da_attention_lambda) * loss_da
        return loss

    def _compute_da_loss(self, encoded_history):
        # fully-connected layer
        self.da_logits = tf.layers.dense(
                encoded_history, 
                self.hparams.num_da_classes, name="logits"
        ) # [batch_size, num_da_classes]

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.da_labels,
                logits=self.da_logits)

        return tf.reduce_mean(losses), tf.arg_max(self.da_logits, -1)

    @classmethod
    def load(cls, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        var_list = {}

        for var in saver_variables:
            if var.op.name[:4] == "asr/":
                var_list[var.op.name[4:]] = var
                #print(var.op.name[4:])
        #for var in saver_variables:
        #    if not (var.op.name[:4] == "asr/" and (var.op.name[-4:] == "Adam" or var.op.name[-6:] == "Adam_1")):
        #        print(var.op.name)
        #        var_list[var.op.name] = var
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
    
    def output_result(self, ground_truth_labels, predicted_labels, extra_ops):
        for gt_text, gt_da, pr_text, pr_da, extra in zip(ground_truth_labels[0], ground_truth_labels[1],
                predicted_labels[0], predicted_labels[1], extra_ops[0]):
            gt_text = [str(id) for id in gt_text if id < self.hparams.num_classes - 2]
            pr_text = [str(id) for id in pr_text if id < self.hparams.num_classes - 2]
            with open(self.hparams.result_output_file, "a") as f:
                f.write('\t'.join([
                    ' '.join(gt_text),
                    ' '.join(pr_text),
                    str(gt_da),
                    str(pr_da),
                    ' '.join([str(f) for f in extra])
                ]) + '\n')
