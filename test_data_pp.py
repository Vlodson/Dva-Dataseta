import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy import signal
import os

images = []

for i in os.listdir("all_img_test"):
    images.append(np.asarray(Image.open("all_img_test/{}".format(i)).convert('LA')))

images = np.array(images)
images = images[:,:,:,0]
images = images / np.max(images)
#---

pre_fft_sounds = []
sound_labels = []

for i in os.listdir("all_snd_test"):
    fs, data = wavfile.read("all_snd_test/{}".format(i))
    pre_fft_sounds.append(data)

pre_fft_sounds = np.array(pre_fft_sounds)
#===============================================================================

def label(path, label, arr):
    for i in os.listdir("Img_data/Test/{}".format(path)):
        arr.append(label)

image_labels = []

label("cello", [1,0,0,0], image_labels)
label("flute", [0,1,0,0], image_labels)
label("sax", [0,0,1,0], image_labels)
label("violin", [0,0,0,1], image_labels)
#---

def labels(path, label, arr):
    for i in os.listdir("Snd_data/Training/{}".format(path)):
        arr.append(label)

labels("cello", [1,0,0,0], sound_labels)
labels("flute", [0,1,0,0], sound_labels)
labels("sax", [0,0,1,0], sound_labels)
labels("violin", [0,0,0,1], sound_labels)
#===============================================================================

pre_fft_sounds = pre_fft_sounds.reshape(pre_fft_sounds.shape[0], pre_fft_sounds.shape[2], pre_fft_sounds.shape[1])

sounds = []

for i in range(pre_fft_sounds.shape[0]):
    sounds.append(signal.spectrogram(pre_fft_sounds[i])[2])

sounds = np.array(sounds)
sounds = sounds.reshape(sounds.shape[0], sounds.shape[1]*sounds.shape[2], sounds.shape[3])

sounds = (sounds - np.min(sounds)) / (np.max(sounds) - np.min(sounds))
#===============================================================================
