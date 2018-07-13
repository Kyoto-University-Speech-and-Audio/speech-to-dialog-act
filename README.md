Experimental speech recognition library

# Programs

- `train.py`
- `eval.py`
- `infer.py`
- `infer_online.py`

# Folders

- `saved_models/<model_name>`: Each checkpoint is saved as `csp.<tag>.ckpt`. Load a pretrained model by specifying `<tag>`
- `log/<model_name>`: Log folder for tensorboard. Launch tensorboard by running `tensorboard --logdir=log`

# Preparing Data

Default loader will load data from one single file with following syntax

```
<path_to_audio_file> <sequence_of_target_labels>
```

where `<path_to_audio_file>` can by `wav`, `htk` or `npy` file that contains original sound / pre-processed acoustic features and `<sequence_of_target_labels>` contains ground-truth labels.

If you use `wav`, path to HTK Speech Recognition Toolkit binary file must be provided in source code.

A vocabulary files must also be prepared with each line having syntax as `<word> <id>`. See below for model configurations.

# Training Examples

## RNN with CTC loss

```
python -m csp.train --config=ctc --dataset=aps
```

## seq2seq with attention mechanism

```
python -m csp.train --config=attention_char --dataset=aps
# or
python -m csp.train --config=attention_word --dataset=aps
```

# Inference

Online inference

```
python -m csp.infer_online --config=attention_aps_char
```

This command will launch a program on terminal which listens to microphone and decode each utterance separated by non-voice range. Used to check accuracy in practice.

# Training & Evaluation Arguments

## Model Configuration

`--config`: Name of the json config files (placed in `model_configs`) containing all hyper parameters and options for training and evaluating model. An example of config file:

```
{
  "name": "attention_word",
  "model": "attention",
  "dataset": "aps",
  "input_unit": "word",
  "vocab_file": "data/aps/words.txt",
  "train_data": "data/aps/train.txt",
  "eval_data": "data/aps/test.txt",
  "encoding": "eucjp",
  "eos_index": 1,
  "sos_index": 2,

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

`--reset`: Model will be trained from scratch regardless of it's previous trained values (remember to backup your trained files). 

`--load` (default: None (latest)): Specified file with tag will be loaded instead of last saved state. For example, `--load=epoch9` will load the model after 9 epochs. Here you can load your own backup model. Tag is included in file name.

`--transfer`: Load parameters from different model (which is placed in corresponding folder and load with `--load`). If specified, a transfering load method must be defined inside model class (`@classmethod def load(cls, sess, ckpt, flags)`). It will use this method instead of default loader. Here you can choose parameters to serialize or initialize additional parameters

`--saved_steps` (default: 300): Save model after specified number of steps

## Training

`--batch_size` (default: 32)

`--gpus`: Use `MultiGPUTrainer` instead of `Trainer`. `MultiGPUTrainer` stores parameter values on CPU and calculate loss on GPUs before combining results for each step.

`--shuffle`: Shuffle data after each epoch

## Evaluation

`--eval` (default: 1000): accuracy on test set will be calculated and written to tensorboard after each specified number of steps. Useful for checking for performance or attention alignments

## Others

`--verbose`: print debug information

`--debug`: run with tensorflow's debug mode

# Customizing for experiments

New model should be subclassed from `BaseModel`, which handles loading hyper-parameters.

`AttentionModel` is highly customizable. You can implement different types of encoder/decoder, attention mechanism or integrate additional parts or values by specifying your functions in initializing method or override existing methods. Some examples can be found in same folder.

# Scripts

To run trainer in background

`./bin/train` will run trainer in background with nohup, which means your process will not be attached to terminal window

Example:

```
./bin/train 0,1 --config=attention_swb_word --gpus
```

will train SwitchBoard speech corpus on two GPUs 0 and 1

# Result on large-scale dataset

Attention model achieved state-of-the-art result for end-to-end model on different world-class datasets.
