from models.attention import AttentionModel
from models.private.da import Model as DAModel

import tensorflow as tf
import numpy as np

from utils import ops_utils, model_utils

class Model(AttentionModel, DAModel):
    default_params = {
        **AttentionModel.default_params,
        **DAModel.default_params,
        "joint_training": False
    }

    def __init__(self):
        AttentionModel.__init__(self, force_alignment_history=True)

    def __call__(self, hparams, mode, iterator, **kwargs):
        super(AttentionModel, self).__call__(hparams, mode, iterator, **kwargs)
        return self
    
    def _assign_input(self):
        self.dlg_ids, \
            (self.inputs, self.input_seq_len), \
            (self.targets, self.target_seq_len), \
            self.da_labels = self.iterator.get_next()
    
    def get_ground_truth_label_placeholder(self): return [self.targets, self.da_labels]

    def get_predicted_label_placeholder(self): return [self.sample_id, self.predicted_da_labels]

    def get_ground_truth_label_len_placeholder(self): return [self.target_seq_len, tf.constant(1)]

    def get_predicted_label_len_placeholder(self): return [self.final_sequence_lengths, tf.constant(1)]

    def get_decode_fns(self):
        return [
            lambda d: self._batched_input.decode(d),
            lambda d: self._batched_input.decode_da(d)
        ]

    def get_da_inputs(self):
        if self.hparams.da_input == "attention_context":
            attention = tf.transpose(self.final_context_state.alignment_history.stack(), [1, 0, 2]) # [batch_size * target_max_len * T]
            attention = tf.expand_dims(attention, -1)
            encoder_outputs = tf.expand_dims(self.encoder_outputs, 1)
            da_inputs = tf.reduce_mean(
                tf.multiply(
                    attention,
                    encoder_outputs,
                ),
                axis=2
            )
            da_inputs = tf.layers.dense(da_inputs, self.hparams.embedding_size)
            da_input_len = self.final_sequence_lengths
        elif self.hparams.da_input == "context_decoder":
            attention = tf.transpose(self.final_context_state.alignment_history.stack(), [1, 0, 2]) # [batch_size * target_max_len * T]
            attention = tf.expand_dims(attention, -1)
            encoder_outputs = tf.expand_dims(self.encoder_outputs, 1)
            da_inputs1 = tf.reduce_mean(
                tf.multiply(
                    attention,
                    encoder_outputs,
                ),
                axis=2
            )
            if self.hparams.attention_layer_size != self.hparams.embedding_size:
                da_inputs1 = tf.layers.dense(da_inputs, self.hparams.embedding_size)
            da_inputs2 = self.decoder_outputs
            if self.hparams.decoder_num_units != self.hparams.embedding_size:
                da_inputs2 = tf.layers.dense(da_inputs, self.hparams.embedding_size)
            da_inputs = tf.concat([da_inputs1, da_inputs2], -1)
            da_input_len = self.final_sequence_lengths
        elif self.hparams.da_input == "decoder_output":
            da_inputs = self.decoder_outputs
            if self.hparams.embedding_size != self.hparams.decoder_num_units:
                da_inputs = tf.layers.dense(da_inputs, self.hparams.embedding_size)
            da_input_len = self.final_sequence_lengths
        elif self.hparams.da_input == "share_embedding":
            da_inputs1 = self.decoder_outputs
            da_inputs1 = tf.layers.dense(da_inputs1, self.hparams.embedding_size / 2)
            da_inputs2 = self.decoder_emb_layer(tf.one_hot(self.sample_id, self.hparams.vocab_size))
            da_inputs2 = tf.layers.dense(da_inputs2, self.hparams.embedding_size / 2)
            da_inputs = tf.concat([da_inputs1, da_inputs2], -1)
            da_input_len = self.final_sequence_lengths
        elif self.hparams.da_input == "predicted_text":
            da_inputs = tf.one_hot(self.sample_id, self.hparams.vocab_size)
            da_inputs = tf.layers.dense(da_inputs, self.hparams.embedding_size)
            da_input_len = self.final_sequence_lengths
        elif self.hparams.da_input == "ground_truth":
            if self.train_mode:
                da_inputs = tf.one_hot(self.targets, self.hparams.vocab_size)
                da_inputs = tf.layers.dense(da_inputs, self.hparams.embedding_size)
                da_input_len = self.target_seq_len
            else:
                da_inputs = tf.one_hot(self.sample_id, self.hparams.vocab_size)
                da_inputs = tf.layers.dense(da_inputs, self.hparams.embedding_size)
                da_input_len = self.final_sequence_lengths
        elif self.hparams.da_input == "combined_attention":
            attention = tf.transpose(self.final_context_state.alignment_history.stack(), [1, 0, 2]) # [batch_size * target_max_len * T]
            attention = tf.expand_dims(attention, -1)
            encoder_outputs = tf.expand_dims(self.encoder_outputs, 1)
            da_inputs1 = tf.reduce_mean(
                tf.multiply(
                    attention,
                    encoder_outputs,
                ),
                axis=2
            )
            da_inputs1 = tf.layers.dense(da_inputs1,
                    self.hparams.embedding_size)
            da_input_len = self.final_sequence_lengths
                
            da_inputs2 = tf.one_hot(self.sample_id, self.hparams.vocab_size)
            da_inputs2 = tf.layers.dense(da_inputs2, self.hparams.embedding_size)
            #da_inputs2 = tf.one_hot(self.sample_id, self.hparams.vocab_size)
            #da_inputs2 = tf.layers.dense(da_inputs2,
            #        self.hparams.embedding_size / 2)
            # da_inputs = tf.concat([da_inputs1, da_inputs2], -1)
            da_inputs = da_inputs1 + da_inputs2
        elif self.hparams.da_input == "combined_decoder_output":
            da_inputs1 = self.decoder_outputs
            da_inputs1 = tf.layers.dense(da_inputs1,
                    self.hparams.embedding_size / 2)
            da_input_len = self.final_sequence_lengths
            da_inputs2 = tf.one_hot(self.sample_id, self.hparams.vocab_size)
            da_inputs2 = tf.layers.dense(da_inputs2,
                    self.hparams.embedding_size / 2)
            da_inputs = tf.concat([da_inputs1, da_inputs2], -1)
        elif self.hparams.da_input == "combined_decoder_output_sum":
            da_inputs1 = self.decoder_outputs
            da_inputs1 = tf.layers.dense(da_inputs1,
                    self.hparams.embedding_size)
            da_input_len = self.final_sequence_lengths
            da_inputs2 = tf.one_hot(self.sample_id, self.hparams.vocab_size)
            da_inputs2 = tf.layers.dense(da_inputs2,
                    self.hparams.embedding_size)
            da_inputs = tf.concat(da_inputs1 + da_inputs2, -1)

        return da_inputs, da_input_len

    def _build_graph(self):
        with tf.variable_scope("asr"):
            loss_asr = AttentionModel._build_graph(self)
        
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
    def load(cls, sess, ckpt, transfer):
        #super().load(sess, ckpt, flags)
        #return
        saver_variables = tf.global_variables()
        var_list = {}

        # transfer from ASR
        if True: # transfer == "asr":
            for var in saver_variables:
                if var.op.name[:4] == "asr/":
                    var_list[var.op.name[4:]] = var
        
        # transfer from partly-trained model
        if False: # transfer == "frozen_asr":
            ignore_vars = [
                "asr/decoder/attention_wrapper/attention_layer/kernel/Adam"
                "asr/decoder/attention_wrapper/attention_layer/kernel/Adam_1"
            ]
            for var in saver_variables:
                if var.op.name[:11] == "asr/decoder" \
                        and (var.op.name[-4:] == "Adam" or var.op.name[-6:] == "Adam_1"):
                    continue
                var_list[var.op.name] = var

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
    
    def output_result(self, ground_truth_labels, predicted_labels,
            ground_truth_label_len, predicted_label_len, extra_ops, eval_count):
        with open(self.hparams.result_output_file, "a") as f:
            for gt_text, gt_da, pr_text, pr_da, extra in zip(ground_truth_labels[0], ground_truth_labels[1],
                    predicted_labels[0], predicted_labels[1], extra_ops[0]):
                gt_text = [str(id) for id in gt_text if id <
                        self.hparams.vocab_size - 2]
                pr_text = [str(id) for id in pr_text if id <
                        self.hparams.vocab_size - 2]
                print(self.hparams.result_output_file)
                f.write('\t'.join([
                    #' '.join(gt_text),
                    #' '.join(pr_text),
                    str(gt_da),
                    str(pr_da),
                    ' '.join([str(f) for f in extra])
                ]) + '\n')
