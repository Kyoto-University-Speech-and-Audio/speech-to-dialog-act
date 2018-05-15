import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

def wav_to_features(filenames_dataset, hparams, feature_count):
    dataset = filenames_dataset.map(lambda filename: io_ops.read_file(filename))
    dataset = dataset.map(lambda wav_loader: contrib_audio.decode_wav(wav_loader, desired_channels=1))
    dataset = dataset.map(lambda wav_decoder:
                                  (contrib_audio.audio_spectrogram(
                                      wav_decoder.audio,
                                      window_size=int(hparams.sample_rate * hparams.window_size_ms / 1000),
                                      stride=int(hparams.sample_rate * hparams.window_stride_ms / 1000),
                                      magnitude_squared=True), wav_decoder.sample_rate))
    dataset = dataset.map(lambda spectrogram, sample_rate: contrib_audio.mfcc(
        spectrogram, sample_rate,
        dct_coefficient_count=feature_count))
    dataset = dataset.map(lambda inputs: (
        inputs,
        tf.nn.moments(inputs, axes=[1])
    ))
    dataset = dataset.map(lambda inputs, moments: (
        tf.divide(tf.subtract(inputs, moments[0]), moments[1]),
        tf.shape(inputs)[1]
    ))
    dataset = dataset.map(lambda inputs, seq_len: (
        inputs[0],
        seq_len
    ))
    return dataset