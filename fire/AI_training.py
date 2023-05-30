from IPython import display
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from numba import cuda
from functions import *


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# define et retrieve the dataset path
DATASET_PATH = 'dataset'
data_dir = pathlib.Path(DATASET_PATH)
print(data_dir)

# retrieve and print the labels
commands = np.delete(np.array(tf.io.gfile.listdir(str(data_dir))), 0)
print('Commands:', commands)

# retrieve the filenames and shuffle them
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)


print('Number of total examples:', num_samples)
print('Number of fire examples:',
      len(tf.io.gfile.listdir(str(data_dir/commands[1]))))
print('Number of other examples:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

# define the training/validation/test datasets lengths
L_train = 0.6
L_test = 0.2
L_val = 0.2

# define the filenames of each set
train_files = filenames[:int(L_train*num_samples)]
val_files = filenames[int(L_train*num_samples):int((L_train+L_val)*num_samples)]
test_files = filenames[-int(L_test*num_samples):]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

# AUTOTUNE is a parameter used to set the number of parallel calls dynamically depending on available CPU
AUTOTUNE = tf.data.AUTOTUNE

# creates a TensorFlow dataset based on the training set
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
# apply the function 'get_waveform_and_label' to each file of the training set
waveform_ds = files_ds.map(
    map_func=get_waveform_and_label,
    num_parallel_calls=AUTOTUNE)

# apply the function 'get_spectrogram' to each file of the training set
for waveform, label in waveform_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectrogram = get_spectrogram(waveform)
print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape)
print('Audio playback')
display.display(display.Audio(waveform, rate=16000))

# apply the function 'get_spectrogram_and_label_id' to each file of the training set
spectrogram_ds = waveform_ds.map(
    map_func=get_spectrogram_and_label_id,
    num_parallel_calls=AUTOTUNE)

# uses the function 'preprocess_dataset' to do the same operations on the other sets
train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

for spectrogram, _ in spectrogram_ds.take(1):
    input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)


# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

# defines the different layers of the model
model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

# print a summary of the neural network
model.summary()

# configures the model
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# defines the number of data fitting over the entire dataset
EPOCHS = 20
# data fitting with the training and validation sets
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)


###########################
####### PERFORMANCE #######
###########################

# retrieves test files and corresponding labels
test_audio = []
test_labels = []
for audio, label in test_ds:
    test_audio.append(audio.numpy())
    test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

# realizes the prediction on the test set
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

# compares the predictions with the truth and prints the Accuracy
test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

### CONFUSION MATRIX ###
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=commands,
            yticklabels=commands,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

# saves the model in the 'model' directory
model.save('model')

device = cuda.get_current_device()
device.reset()