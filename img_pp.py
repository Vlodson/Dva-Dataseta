""" Ova skripta sluzi za ucitavanje, pravljenje labela i
pravljenje trening seta sa nasumicno rasporedjenim trening slikama + iz trening
slika pravljenje seta za trening testiranja i pravljenje test seta (ovo poslednje je mozda)
1 je za cello, 2 je sax, 3 je violin """

import numpy as np
from PIL import Image
import os


# sa listdir idem u folder location, sa i idem kroz iteme u folderu i svaki
# appendujem na images (odnosno svaku sliku dodajem na listu slika)
images = []

for i in os.listdir("all_img"):
    images.append(np.asarray(Image.open("all_img/{}".format(i)).convert('LA')))

images = np.array(images)
images = images[:,:,:,0]
images = images / np.max(images)

#===============================================================================


# pravim listu labela tako sto za svaki item iz foldera insturmenta appendujem
# labelu na listu
def label(path, label, arr):
    for i in os.listdir("Img_data/Training/{}".format(path)):
        arr.append(label)

image_labels = []

label("cello", [1,0,0,0], image_labels)
label("flute", [0,1,0,0], image_labels)
label("sax", [0,0,1,0], image_labels)
label("violin", [0,0,0,1], image_labels)

#===============================================================================

# shuffleujem slike i njihove labele zbog boljeg ucenja CNN-a, obicnom zamenom
# dva random elementa
# for i in range(len(images)):
#     rand = np.random.randint(0, len(images))
#
#     temp = images[i]
#     images[i] = images[rand]
#     images[rand] = temp
#
#     temp = image_labels[i]
#     image_labels[i] = image_labels[rand]
#     image_labels[rand] = temp

#===============================================================================
