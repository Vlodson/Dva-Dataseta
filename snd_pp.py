import numpy as np
from scipy.io import wavfile
import os


sounds = []
sound_labels = []

for i in os.listdir("Snd_data/Training/all"):
    fs, data = wavfile.read("Snd_data/Training/all/{}".format(i))
    sounds.append(data)

sounds = np.array(sounds)


def labels(path, label, arr):
    for i in os.listdir("Snd_data/Training/{}".format(path)):
        arr.append(label)

labels("cello", 1, sound_labels)
labels("flute", 2, sound_labels)
labels("sax", 3, sound_labels)
labels("violin", 4, sound_labels)

for i in range(len(sounds)):
    rand = np.random.randint(0, len(sounds))

    temp = sounds[i]
    sounds[i] = sounds[rand]
    sounds[rand] = temp

    temp = sound_labels[i]
    sound_labels[i] = sound_labels[rand]
    sound_labels[rand] = temp

for i in sounds:
    i = np.fft.fft2(i)
