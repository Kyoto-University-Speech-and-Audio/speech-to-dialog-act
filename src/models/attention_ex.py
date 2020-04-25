from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib as mpl
from six.moves import xrange as range

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.seq2seq import BeamSearchDecoder

from .attention import AttentionModel as BaseModel
from .attentions.location_based_attention import LocationBasedAttention
from utils import model_utils
from .helpers.basic_context_decoder import BasicContextDecoder


class AttentionExModel(BaseModel):
    def __call__(self, hparams, mode, batched_input, **kwargs):
        BaseModel.__call__(self, hparams, mode, batched_input, **kwargs)

        if self.infer_mode or self.eval_mode:
            attention_summary = self._get_attention_summary()
            if attention_summary is not None:
                self.summary = tf.summary.merge([attention_summary])
            else:
                self.summary = tf.no_op()

        return self

    def get_extra_ops(self):
        if self.hparams.output_type == 'beam_scores':
            return [self.beam_scores]
        elif self.hparams.output_type == 'sample_ids':
            return []
        return [self.encoder_outputs, self.encoder_final_state]
        return []
        return [self.decoder_outputs]

        attention = tf.transpose(self.final_context_state.alignment_history.stack(),
                                 [1, 0, 2])  # [batch_size * target_max_len * T]
        attention = tf.expand_dims(attention, -1)
        encoder_outputs = tf.expand_dims(self.encoder_outputs, 1)
        da_inputs = tf.reduce_mean(
            tf.multiply(
                attention,
                encoder_outputs,
            ),
            axis=2
        )
        return [da_inputs]
        # return tf.no_op()

    def _get_attention_summary(self):
        return None
        self.attention_images = tf.no_op()
        if self.hparams.beam_width > 0: return None
        attention_images = self.final_context_state.alignment_history.stack()  # max_len * batch_size * T
        attention_images = tf.transpose(attention_images, [1, 0, 2])
        self.attention_images = attention_images

        return None
        inferred_labels = self.sample_id[0, :]
        indices = tf.where(tf.not_equal(inferred_labels, tf.constant(1, inferred_labels.dtype)))
        attention_images = attention_images[:self.input_seq_len[0], :tf.shape(indices)[0]]
        attention_images = tf.expand_dims(attention_images, 0)
        return None
        attention_images = tf.expand_dims(tf.transpose(attention_images, [0, 2, 1]),
                                          -1)  # batch_size * max_len * # T * 1
        attention_images = 1 - attention_images
        attention_images *= 255
        attention_summary = tf.summary.image("attention_images", attention_images, max_outputs=1)
        return attention_summary

    @classmethod
    def load(cls, sess, ckpt, flags):
        saver_variables = tf.global_variables()
        var_list = {}
        if False:
            for var in saver_variables:
                var_list[var.op.name] = var
                if var.op.name in ["asr/decoder/beam_outputs/memory_layer/kernel"]:
                    del var_list[var.op.name]
            #    if var.op.name[:7] == "encoder":
            #        var_list[var.op.name] = var
            sess.run([
                tf.assign(sess.graph.get_tensor_by_name("asr/decoder/beam_outputs/memory_layer/kernel:0"),
              
                    sess.graph.get_tensor_by_name("asr/decoder/memory_layer/kernel:0"))
            ])
        if True:
            for var in saver_variables:
                if var.op.name in ['Variable', 'Variable_1', 'batch_size',
                        'eval_batch_size']:
                    var_list[var.op.name] = var
                else:
                    var_list['asr/' + var.op.name] = var
        
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt)


    def output_result(self, ground_truth_labels, predicted_labels,
                      ground_truth_label_len, predicted_label_len, extra_ops, eval_count):
        target_ids = ground_truth_labels[0]
        sample_ids = predicted_labels[0]
        ex1s = extra_ops[0]
        with open(self.hparams.result_output_file, "a") as f:
            for ids1, ids2, ex1 in zip(target_ids, sample_ids,
                                       ex1s):
                _ids1 = [str(id) for id in ids1 if id < self.hparams.vocab_size - 2]
                _ids2 = [str(id) for id in ids2 if id < self.hparams.vocab_size - 2]
                fn = "%s/%d.npy" % (self.hparams.result_output_folder, eval_count)

                if self.hparams.output_type == 'beam_scores':
                    f.write('\t'.join([
                        # filename.decode(),
                        ' '.join([str(x) for x in ex1[-1]])
                        # ' '.join(_ids1),
                        # ' '.join(_ids2),
                        # ' '.join(self._batched_input_test.decode(ids2)),
                        # fn
                    ]) + '\n')
                elif self.hparams.output_type == 'sample_ids':
                    f.write('\t'.join([
                        # filename.decode(),
                        ' '.join(_ids1),
                        ' '.join(_ids2),
                        # ' '.join(self._batched_input_test.decode(ids2)),
                        # fn
                    ]) + '\n')
                # att = att[:len(_ids2), :]
                # np.save(fn, att)
