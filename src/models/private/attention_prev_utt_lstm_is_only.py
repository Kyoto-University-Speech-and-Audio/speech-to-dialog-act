import tensorflow as tf
from .attention_prev_utt_lstm import AttentionModel as BaseAttentionModel
NUM_SPEAKERS = 2

class AttentionModel(BaseAttentionModel):
    def __init__(self):
        super().__init__(
            feed_initial_state=True,
            use_context_vector=False
        )