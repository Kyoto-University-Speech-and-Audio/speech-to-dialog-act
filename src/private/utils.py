
def get_model_class(model_name):
    Model = None
    if model_name == 'attention_context':
        from .models.attention_context import AttentionModel
        return AttentionModel
    elif model_name == 'attention_e2e':
        from models.private.da_attention_e2e import AttentionE2EModel
        return AttentionE2EModel
    elif model_name == 'attention_e2e2':
        from models.private.da_attention_e2e2 import AttentionE2EModel
        return AttentionE2EModel
    elif model_name == 'da_attention_e2e':
        from models.private.da_attention_e2e import Model
        return Model
    elif model_name == 'da_attention_e2e2':
        from models.private.da_attention_e2e2 import Model
        return Model
    elif model_name == 'attention_prev_utt':
        from .models.attention_prev_utt_lstm import AttentionModel
        return AttentionModel
    elif model_name == 'attention_prev_utt_is':
        from .models.attention_prev_utt_lstm_is import AttentionModel
        return AttentionModel
    elif model_name == 'attention_prev_utt_is_only':
        from .models.attention_prev_utt_lstm_is_only import AttentionModel
        return AttentionModel
    elif model_name == 'ctc-attention':
        from .models.ctc_attention import CTCAttentionModel as Model
    elif model_name == 'encoder_speaker_change':
        from .models.encoder_speaker_change import Model
    elif model_name == 'da':
        from models.private.da import Model
    elif model_name == 'da_seg':
        from models.private.da_seg import Model
    elif model_name == 'da_attention':
        from models.private.da_attention import Model
    elif model_name == 'da_attention_seg':
        from models.private.da_attention_seg import Model
    elif model_name == 'da_utt_attention':
        from models.private.da_utt_attention import Model
    return Model


def get_batched_input_class(dataset):
    BatchedInput = None

    if dataset == 'swbd_seg':
        from datasets.private.swbd_seg import BatchedInput
    elif dataset == 'swda':
        from datasets.private.swda import BatchedInput
    elif dataset == 'swda_seg':
        from datasets.private.swda_seg import BatchedInput
    elif dataset == 'swbd_order':
        from .datasets.private.swbd_order import BatchedInput
    elif dataset == 'swbd_order_speaker_change':
        from .datasets.private.swbd_order_speaker_change import BatchedInput
    elif dataset == 'aps-word':
        from .datasets.private.csj import BatchedInput
    elif dataset == 'erato':
        from .datasets.private.erato import BatchedInput
    elif dataset == 'erato_context':
        from .datasets.private.erato_context import BatchedInput
    elif dataset == 'erato_prev_utt':
        from .datasets.private.erato_prev_utt import BatchedInput
    return BatchedInput
