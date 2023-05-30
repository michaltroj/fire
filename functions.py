import tensorflow as tf
import numpy as np
import os

def decode_audio(audio_binary):
    """
    Decode WAV-encoded audio files to `float32` tensors, normalized to the [-1.0, 1.0] range.

    Parameters
    ----------
    audio_binary : tf.Tensor of type string
        -- tensor
    
    Returns
    -------
    audio : tf.Tensor of type float32
        -- tensor
    
    """
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
    """
    Retrieve the label of an audio file.

    Parameters
    ----------
    file_path : str of tf.Tensor of type string
        -- name of the file path

    Returns
    -------
    parts : tf.Tensor of type string
        -- name of the file's label
    
    """
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]

def get_waveform_and_label(file_path):
    """
    Gets the waveform and the label of a WAV file.

    Parameters
    ----------
    file_path : str of tf.Tensor of type string
        -- name of the file path

    Returns
    -------
    waveform : tf.Tensor of type float32
        -- vector of wave points
    label : tf.Tensor of type string
        -- name of the file's label
    
    """
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_spectrogram(waveform):
    """
    Gets the spectrogram of a WAV file.

    Parameters
    ----------
    waveform : tf.Tensor of type float32
        -- vector of wave points

    Returns
    -------
    spectrogram : tf.Tensor of type float32
        -- tensor of the modules of short-time Fourier transform values
    
    """
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 80000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [80000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    """
    Gets the spectrogram and the label id of a WAV file.

    Parameters
    ----------
    audio : tf.Tensor of type float32
        -- vector of wave points
    label : str or tf.Tensor of type string
        -- name of the file's label

    Returns
    -------
    spectrogram : tf.Tensor of type float32
        -- tensor of the modules of short-time Fourier transform values
    label_id : int
        -- 0 or 1
    
    """
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == ['other', 'fire'])
    return spectrogram, label_id

def preprocess_dataset(files):
    """
    Preprocesses the dataset files for learning.

    Parameters
    ----------
    files : tf.Tensor of type string
        -- names of the files to be processed

    Returns
    -------
    output_ds : tf.Tensor of type float32 and int (0 or 1)
        -- tensor of the spectrograms and corresponding labels
    
    """
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=tf.data.AUTOTUNE)
    output_ds = output_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=tf.data.AUTOTUNE)
    return output_ds

def plot_spectrogram(spectrogram, ax):
    """
    Displays a spectrogram.

    Parameters
    ----------
    spectrogram : tf.Tensor of type float32
        -- tensor of the modules of short-time Fourier transform values
    ax : tf.tensor of np.array
        -- axes of the spectrogram
    
    """
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

# This function returns a list with the filesnames of a path, and removes the hidden files
def listdir_nohidden(path):
    """
    Retrieves a list with the filesnames of a path, and removes the hidden files.

    Parameters
    ----------
    path : str
        -- path name

    Returns
    -------
    files : list
        -- names of the files in the directory 
    
    """
    files = os.listdir(path)
    index = [i for i in range(len(files)) if files[i][0] == '.']
    [files.pop(i) for i in range(len(index))]
    return files