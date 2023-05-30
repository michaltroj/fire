from __future__ import print_function
import wave
from pydub import AudioSegment
import scipy.io.wavfile as wavfile
import os
from functions import *

# Ask the user which path to consider
path_base = input("Dans quel dossier sont les fichiers audio de base ? : ")
path_split = input(
    "Dans quel dossier les nouveaux fichiers audio doivent être placés ? : ")
files = listdir_nohidden(path_base)

# Spliting the files
for i in range(len(files)):

    file_name = os.path.join(path_base, files[i])
    fs_rate, signal = wavfile.read(file_name)
    N = signal.shape[0]
    secs = N / float(fs_rate)
    number_sample = int(secs//5)
    values = [5*i for i in range(number_sample)]
    Data = [0]*(len(values)-1)
    print("Temps : ", secs)
    print("Number_sample = ", number_sample)
    # file to extract the snippet from
    with wave.open(file_name, "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        for k in range(len(values)-1):
            # set position in wave to start of segment
            infile.setpos(int(values[k] * framerate))
            # extract data
            Data[k] = infile.readframes(
                int((values[k+1] - values[k]) * framerate))
            print("Valeur de data : ", Data[k][len(values)-1])

    print('done getting data : ' + str(i))
    for m in range(number_sample-1):
        # write the extracted data to a new file
        fullname = 'sample_' + str(i) + '_' + str(m) + '.wav'
        with wave.open(os.path.join(path_split, fullname), 'w') as outfile:
            outfile.setnchannels(nchannels)
            outfile.setsampwidth(sampwidth)
            outfile.setframerate(framerate)
            outfile.setnframes(int(len(Data[m]) / sampwidth))
            outfile.writeframes(Data[m])
            print('Enregistrement numero ' + str(i) +
                  'echantillon numero' + str(m) + ' ecrit !')
        sound = AudioSegment.from_wav(os.path.join(path_split, fullname))
        sound = sound.set_channels(1)
        sound.export(os.path.join(path_split, fullname), format="wav")
