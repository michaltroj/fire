from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as ftt
import numpy as np
import argparse
from matplotlib import pyplot as plt

# adds the audio filename as an argument of the script
parser = argparse.ArgumentParser(description='Detect fire')
parser.add_argument('filename', type=str,
                    help='audio file on which we want to perform the frequency analysis')
args = parser.parse_args()
filename = args.filename

fs_rate, signal = wavfile.read(filename)
print("Frequency sampling", fs_rate)
l_audio = len(signal.shape)
print("Channels", l_audio)
if l_audio == 2:
    signal = signal.sum(axis=1) / 2
N = signal.shape[0]
print("Complete Samplings N", N)
secs = N / float(fs_rate)
print("secs", secs)
Ts = 1.0/fs_rate  # sampling interval in time
print("Timestep between samples Ts", Ts)
# time vector as scipy arange field / numpy.ndarray
t = np.arange(0, secs, Ts)
FFT = abs(ftt.fft(signal))
print(FFT)
FFT_side = FFT[range(N//2)]  # one side FFT range
freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(N//2)]  # one side frequency range
fft_freqs_side = np.array(freqs_side)
plt.subplot(211)
p1 = plt.plot(t, signal, "g")  # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(212)
# p2 = plt.plot(freqs, FFT, "r")  # plotting the complete fft spectrum
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Count dbl-sided')
# plt.subplot(313)
# plotting the positive fft spectrum
p3 = plt.plot(freqs_side, abs(FFT_side), "b")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')
plt.show()
