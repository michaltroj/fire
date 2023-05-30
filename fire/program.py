import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
from functions import *

# adds the audio filename as an argument of the script using the argparse module
parser = argparse.ArgumentParser(description='Detect fire')
parser.add_argument('filename', type=str,
                    help='audio file in which we want to detect whether there is fire or not')
args = parser.parse_args()
filename = args.filename

# the different labels
states = ['other', 'fire']
print(states)


# loading the model stored in 'model' directory
model = load_model('model')
# preprocessing of the dataset
data = preprocess_dataset(np.array([filename]))
print(len(data))

#predicting the category and printing the result
for audio, label in data:
    prediction = model.predict(np.array([audio]))
    result = states[np.argmax(prediction)]
    st = ['#', '']
    print('########' + st[np.argmax(prediction)])
    print('#', result, '#')
    print('########' + st[np.argmax(prediction)])