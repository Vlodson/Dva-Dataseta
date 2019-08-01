import numpy as np
from convo import convo
from maxpool import MP
from full import full_layer
from output import output
from test_data_pp import images, image_labels, sounds, sound_labels
print("podaci ucitani")
import datetime
import matplotlib.pyplot as plt

num_data = images.shape[0] # upises obicno broj, ali ako hoces sve, onda data.shape[0]

images = images[:num_data]
image_labels = image_labels[:num_data]

sounds = sounds[:num_data]
sound_labels = sound_labels[:num_data] # nzm da li treba ovo uopste
#---

labels = 4 # violina, sax, violoncelo, flauta

ly1_img = 8
ly1_snd = 8
ly2 = 10

filter1_img = np.genfromtxt("W_b/Image/filter1_img.csv", delimiter = ',')
filter1_img = (filter1_img.shape[0], filter1_img.shape[1], filter1_img)

filter1_snd = np.genfromtxt("W_b/Sound/filter1_snd.csv", delimiter = ',')
filter1_snd = (filter1_snd.shape[0], filter1_snd.shape[1], filter1_snd)

bias1_img = np.genfromtxt("W_b/Image/bias1_img.csv", delimiter = ',')

bias1_snd = np.genfromtxt("W_b/Sound/bias1_snd.csv", delimiter = ',')
#---

filter2_img = np.genfromtxt("W_b/Image/filter2_img.csv", delimiter = ',')
filter2_img = (filter2_img.shape[0], filter2_img.shape[1], filter2_img)

filter2_snd = np.genfromtxt("W_b/Sound/filter2_snd.csv", delimiter = ',')
filter2_snd = (filter2_snd.shape[0], filter2_snd.shape[1], filter2_snd)

bias2_img = np.genfromtxt("W_b/Image/bias2_img.csv", delimiter = ',')

bias2_snd = np.genfromtxt("W_b/Sound/bias2_snd.csv", delimiter = ',')
#---

W1_img = np.genfromtxt("W_b/Image/W1_img.csv", delimiter = ',')
W1_img = W1_img.reshape(W1_img.shape[0], W1_img.shape[1])

W1_snd = np.genfromtxt("W_b/Sound/W1_snd.csv", delimiter = ',')
W1_snd = W1_snd.reshape(W1_snd.shape[0], W1_snd.shape[1])

W2_img = np.genfromtxt("W_b/Image/W2_img.csv", delimiter = ',')
W2_img = W2_img.reshape(W2_img.shape[0], W2_img.shape[1])

W2_snd = np.genfromtxt("W_b/Sound/W2_snd.csv", delimiter = ',')
W2_snd = W2_snd.reshape(W2_img.shape[0], W2_snd.shape[1])

W2 = np.append(W2_img, W2_snd)
W2 = W2.reshape(W2_img.shape[0]+W2_snd.shape[0], ly2)

Wo = np.genfromtxt("W_b/Wo.csv", delimiter = ',')
Wo = Wo.reshape(Wo.shape[0], Wo.shape[1])
#===============================================================================


labels_img = np.shape(image_labels)[1]
labels_snd = np.shape(sound_labels)[1]
#===============================================================================

print("Start: ", datetime.datetime.now().time())

c1_img = convo(images, filter1_img, (9,9), bias1_img)
c1_img.convolution()

c1_snd = convo(sounds, filter1_snd, (8,8), bias1_snd)
c1_snd.convolution()
#---

m1_img = MP(c1_img.out, (2,2), (1,1))
m1_img.maxpooling()

m1_snd = MP(c1_snd.out, (1,2), (1,2))
m1_snd.maxpooling()
#---

c2_img = convo(m1_img.out, filter2_img, (1,1), bias2_img)
c2_img.convolution()

c2_snd = convo(m1_snd.out, filter2_snd, (2,2), bias2_snd)
c2_snd.convolution()
#---

m2_img = MP(c2_img.out, (2,2), (1,1))
m2_img.maxpooling()

m2_snd = MP(c2_snd.out, (3,3), (2,2))
m2_snd.maxpooling()
#---

f1_img = full_layer(m2_img.out, ly1_img, W1_img) # ovde se vec radi forwardfeed, nema f za to
#f1_img.out[f1_img.out <= 0] = 0
f1_snd = full_layer(m2_snd.out, ly1_snd, W1_snd)
#---

f1_out = np.zeros((images.shape[0], f1_img.out.shape[1] + f1_snd.out.shape[1])) # konkatiniran output proslog sloja

for j in range(images.shape[0]):
    f1_out[j] = np.append(f1_img.out[j], f1_snd.out[j])

f2 = full_layer(f1_out, ly2, W2)

o = output(f2.out, Wo, labels, image_labels)
o.out[o.out <= 0] = 0
#===============================================================================

tru = 0

for i in range(len(image_labels)):
    print(np.where(o.out[i] == np.max(o.out[i]))[0][0], image_labels[i].index(max(image_labels[i])))
    if np.where(o.out[i] == np.max(o.out[i]))[0][0] == image_labels[i].index(max(image_labels[i])):
        tru += 1

print(tru/len(image_labels) * 100)
print("Finish: ", datetime.datetime.now().time())
