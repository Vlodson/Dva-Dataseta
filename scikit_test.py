from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image
import os


def ucitavanje(instrument, path, br_labele):
    inst = []
    inst_label = []

    for i in os.listdir("Img_data/{}/{}".format(path, instrument)):
        inst.append(np.asarray(Image.open("Img_data/{}/{}/{}".format(path, instrument, i))))
        inst_label.append("{}".format(br_labele))

    return np.array(inst), np.array(inst_label)


cello, cello_label = ucitavanje("cello", "Training", 1)
flute, flute_label = ucitavanje("flute", "Training", 2)
sax, sax_label = ucitavanje("sax", "Training", 3)
violin, violin_label = ucitavanje("violin", "Training", 4)


images, kita = ucitavanje("all", "Training", 1)
image_labels = []

for i in range(len(cello)):
    image_labels.append(1)

for i in range(len(flute)):
    image_labels.append(2)

for i in range(len(sax)):
    image_labels.append(3)

for i in range(len(violin)):
    image_labels.append(4)

for i in range(len(images)):
    rand = np.random.randint(0, len(images))

    temp = images[i]
    images[i] = images[rand]
    images[rand] = temp

    temp = image_labels[i]
    image_labels[i] = image_labels[rand]
    image_labels[rand] = temp



clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
pca = PCA(svd_solver='full', n_components='mle')
pca.fit(images)

#images = images.reshape(945, 258*258*3)
#clf.fit(images, image_labels)
