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
        self.dlg_ids, (self.inputs, self.input_seq_len), (self.targets, self.target_seq_len), self.da_labels = self.iterator.get_next()
    
    def get_ground_truth_label_placeholder(self): 
        return [self.targets, self.da_labels]

    def get_predicted_label_placeholder(self): 
        return [self.sample_id, self.predicted_da_labels]

    def _build_history(self, emb, rank=0, dtype=tf.int32):
        """
        Args:
            emb: size [batch_size, ...]
        
        Returns:
            history_targets: size [batch_size, history_size + 1, ...]
            history_seq_len: size [batch_size]
        """
        self.prev_embs = tf.get_variable("prev_embs", [0,
            self.hparams.decoder_num_units], dtype=dtype, trainable=False)
        self.prev_dlg_ids = tf.get_variable("prev_dlg_ids", [0], dtype=tf.string, 
                initializer=tf.constant_initializer('', tf.string),
                trainable=False)

        # filter input from previous batch with same dialog id
        prev_same_dlg_embs = tf.boolean_mask(
                self.prev_embs,
                tf.equal(self.prev_dlg_ids, self.dlg_ids[0])
        )

        # concatenate previous batch with current batch
        full_embs = tf.concat([prev_same_dlg_embs, emb], 0) # [num_utts, feature_size]
        
        # build history data
        def pad_history(t, i, rank):
            ret = t[tf.maximum(i - self.hparams.num_utt_history, 0):i + 1]
            print(ret)
            return (tf.slice(tf.pad(
                    ret, 
                    [[0, tf.maximum(0, self.hparams.num_utt_history - i)]] + [[0, 0]] * (rank - 1)
            ), [0, 0], [self.hparams.num_utt_history + 1,
                self.hparams.encoder_num_units]))


            return tf.pad(
                    ret, 
                    [[0, tf.maximum(0, self.hparams.num_utt_history - i)]] + [[0, 0]] * (rank - 1)
                    )[:, :self.hparams.decoder_num_units]

        start_id = tf.shape(prev_same_dlg_embs)[0]
        end_id = tf.shape(full_embs)[0]
        history_embs = tf.map_fn(
                lambda i: pad_history(full_embs, i, 2), 
                tf.range(start_id, end_id),
                dtype=dtype
        ) # [batch_size, history_len + 1, feature_size]
        history_seq_len = tf.map_fn(
                lambda i: tf.minimum(i + 1, self.hparams.num_utt_history + 1),
                tf.range(start_id, end_id)
        )
        return history_embs, history_seq_len


    def _build_graph(self):
        with tf.variable_scope("asr"):
            loss_asr = AttentionModel._build_graph(self)
        
        with tf.variable_scope("da_recog"):
            history_embs, history_seq_len = self._build_history(
                self.final_context_state.cell_state[0],
                rank=0,
                dtype=tf.float32
            )

            encoded_history = self._build_utt_encoder(history_embs, history_seq_len)

            loss_da, self.predicted_da_labels = self._get_loss(encoded_history)
            with tf.control_dependencies([loss_da]):
                self.update_prev_inputs = self._build_update_prev_inputs(history_embs)
        
        loss = self.hparams.da_attention_lambda * loss_asr + (1 - self.hparams.da_attention_lambda) * loss_da
        return loss
    
    def _build_update_prev_inputs(self, embs):
        return tf.group([
            tf.assign(self.prev_dlg_ids, self.dlg_ids, validate_shape=False),
            tf.assign(self.prev_embs, embs, validate_shape=False),
            #tf.assign(self.prev_speakers, self.speakers)
        ])

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
            return list(filter(lambda var: var.op.name[:11] != "asr/encoder",
                    trainable_vars))
            return trainable_vars
        else:    
            return list(filter(lambda var: var.op.name[:8] == "da_recog",
            trainable_vars))

    def get_extra_ops(self):
        return [self.decoder_outputs]
