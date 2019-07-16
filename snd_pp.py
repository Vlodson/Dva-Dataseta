import numpy as np
from scipy.io import wavfile
from scipy import signal
import os


pre_fft_sounds = []
sound_labels = []

for i in os.listdir("all_snd"):
    fs, data = wavfile.read("all_snd/{}".format(i))
    pre_fft_sounds.append(data)

pre_fft_sounds = np.array(pre_fft_sounds)


def labels(path, label, arr):
    for i in os.listdir("Snd_data/Training/{}".format(path)):
        arr.append(label)

labels("cello", [1,0,0,0], sound_labels)
labels("flute", [0,1,0,0], sound_labels)
labels("sax", [0,0,1,0], sound_labels)
labels("violin", [0,0,0,1], sound_labels)

# for i in range(len(pre_fft_sounds)):
#     rand = np.random.randint(0, len(pre_fft_sounds))
#
#     temp = pre_fft_sounds[i]
#     pre_fft_sounds[i] = pre_fft_sounds[rand]
#     pre_fft_sounds[rand] = temp
#
#     temp = sound_labels[i]
#     sound_labels[i] = sound_labels[rand]
#     sound_labels[rand] = temp

pre_fft_sounds = pre_fft_sounds.reshape(pre_fft_sounds.shape[0], pre_fft_sounds.shape[2], pre_fft_sounds.shape[1])

sounds = []

for i in range(pre_fft_sounds.shape[0]):
    sounds.append(signal.spectrogram(pre_fft_sounds[i])[2])

sounds = np.array(sounds)
sounds = sounds.reshape(sounds.shape[0], sounds.shape[1]*sounds.shape[2], sounds.shape[3])

sounds = (sounds - np.min(sounds)) / (np.max(sounds) - np.min(sounds))
