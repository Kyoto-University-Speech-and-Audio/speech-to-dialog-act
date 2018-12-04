Experimental speech recognition library

# Scripts & Paths

These scripts are for training, evaluation and inference

- `train.py`
- `eval.py`
- `infer.py`
- `infer_online.py`

While training, log and checkpoint files are generated in these folders

- `saved_models/<model_name>`: Each checkpoint is saved as `csp.<tag>.ckpt`. Load a pretrained model by specifying `<tag>`
- `log/<model_name>`: Log folder for tensorboard. Launch tensorboard by running `tensorboard --logdir=log`

# Data Preparation

Default loader loads data from a file with the following syntax (you can define your own inputting method in `src/datasets`)

```
<path_to_audio_file> <sequence_of_target_labels>
```

where `<path_to_audio_file>` can be `wav`, `htk` or `npy` file that contains original sound / pre-processed acoustic features and `<sequence_of_target_labels>` contains ground-truth labels.

If you use `wav`, you have to provide the paths to HTK Speech Recognition Toolkit in `configs.py`.

A vocabulary files must also be prepared with each line containing a label (word or character). See below for model configurations.

# Examples

Train an RNN-based model with CTC loss

```
CUDA_VISIBLE_DEVICES=0,1 python -m src.train --config=ctc_aps_sps
```

Train an seq2seq model with attention mechanism

```
CUDA_VISIBLE_DEVICES=0,1 python -m src.train --config=attention_aps_sps_char
# or
CUDA_VISIBLE_DEVICES=0,1 python -m src.train --config=attention_aps_sps_word
```

Evaluate with last checkpoint

```
python -m src.eval --config=attention_aps_sps_char
```

Online inference

```
python -m csp.infer_online --config=attention_aps_sps_char
```

This command will launch a program on terminal which listens to microphone and decode each utterance separated by non-voice range. Used to check accuracy in real time.

# Training & Evaluation Arguments

## Model Configuration

`--config`: Name of the json config files (placed in `model_configs`) containing all hyper parameters and options for training and evaluating model. An example of config file:

```
{
  "model": "attention",
  "input": "aps",
  "input_unit": "word",
  "vocab_file": "data/aps/words.txt",
  "train_data": "data/aps/train.txt",
  "eval_data": "data/aps/test.txt",
  "encoding": "eucjp",
  "eos_index": 1,
  "sos_index": 2,
  "metrics": "wer"

  "batch_size": 32,
  "beam_width": 0,

  "encoder_type": "bilstm",
  "encoder_num_units": 512,
  "decoder_num_units": 256,
  "attention_num_units": 256,
  "attention_layer_size": 256,
  "attention_energy_scale": true,

  "length_penalty_weight": 0.5,
  ...
}
```

## Load and Save

Default behaviour is training from last saved model. These method can be used for optional load.

`--reset`: Model will be trained from scratch regardless of its saved parameters (existing checkpoint will be overwritten). 

`--load` (default: None (latest)): Checkpoint to be loaded. Eg: `--load=epoch9`, `--load=best_0`

`--transfer`: Load pre-trained parameters from other model. A load method (mapping pre-trained parameters to model's parameters) must be defined (`@classmethod def load(cls, sess, ckpt, flags)`), which will be used instead of default loader.

`--saved_steps` (default: 300): Save model after this number of steps

## Training

`--batch_size` (default: 32)

`--shuffle`: Shuffle data after each epoch

## Evaluation

`--eval` (default: 0): update word error rate on tensorboard after a number of steps. If 0, evaluation is run after each epoch.

## Others

`--verbose`: print debug information

`--debug`: run with tensorflow's debug mode

# Customization for experiments

New model should be subclassed from `BaseModel`, which handles loading hyper-parameters.

`AttentionModel` is highly customizable. You can implement different types of encoder/decoder, attention mechanism or integrate additional components or embeddings by specifying your functions in initializing method or override existing methods. Some examples can be found in same folder.

# Results

Results with sample configurations (not optimal configs):

| Config file | Model | Dataset | Unit | LER |
|-------------|-------|---------|------|-----|
|`ctc_aps_sps.json`|ctc|combined CSJ-ASP & CSJ-SPS|char| - |
|`attention_aps_sps_char.json`|attention-based|combined CSJ-ASP & CSJ-SPS | char | - |
|`attention_aps_sps_word.json`|attention-based|combined CSJ-ASP & CSJ-SPS | word | - |

# Checkpoint

- [x] CTC loss
- [x] Attention mechanism
- [x] Location-based attention
- [ ] Joint CTC-attention
- [ ] Tacotron2

# Others

You can refer to my article for an intuitive explanation on how attention mechanism works

[Sequence to sequence learning with attention mechanism](https://medium.com/@viettrungdang/sequence-to-sequence-learning-with-attention-mechanism-a8964b5e301e)

Most of the datasets are not public, but if you have access to any, you can refer to my pre-processing code in `src/preproc`. The code is written inside a Jupyter Notebook for easy debug but not so well-organized.
