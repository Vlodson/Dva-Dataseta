import numpy as np
from convo import convo
from maxpool import MP
from full import full_layer
from output import output
from img_pp import images, image_labels
print("slike ucitane")
import datetime
import matplotlib.pyplot as plt
import os
#===============================================================================

# randomizuje redosled podataka
# for i in range(len(images)):
#     rand = np.random.randint(0, len(images))
#
#     temp_i = images[i]
#     images[i] = images[rand]
#     images[rand] = temp_i
#
#     temp_i = image_labels[i]
#     image_labels[i] = image_labels[rand]
#     image_labels[rand] = temp_i
#===============================================================================

num_data = 10 # upises obicno broj, ali ako hoces sve, onda data.shape[0]

images = images[:num_data]
image_labels = image_labels[:num_data]

print(image_labels[:5])
#===============================================================================

labels = 4 # violina, sax, violoncelo, flauta
lr = 0.00001 # learn rate, treba da bude mali za pomeranje po slopeovima
learn_iter = 100 # za broj iteracija ucenja
i = 1

# prve random var
filter1_img = (15, 15, np.random.uniform(size = (15,15)))
bias1_img = np.zeros((1,1))

filter2_img = (3, 3, np.random.uniform(size = (3, 3)))
bias2_img = np.zeros((1,1))

# neke const
ly1_img = 10
ly2 = 10
Loss = []
Loss_prev = 1000000 # posto gledam da li je Loss_now manji od Loss_prev,
Loss_now = 1000000 #  ako ih stavim na 0 na pocetku, bice Loss_prev uvek manji
Loss_iter = []

#===============================================================================

print("Start: ", datetime.datetime.now().time())

while i <= learn_iter:

    # za ispis podataka dole
    Loss_prev = Loss_now

    """ FEEDFORWARD """
    c1_img = convo(images, filter1_img, (9,9), bias1_img)
    c1_img.convolution()
    #c1_img.out[c1_img.out <= 0] = 0

    m1_img = MP(c1_img.out, (2,2), (1,1))
    m1_img.maxpooling()

    c2_img = convo(m1_img.out, filter2_img, (1,1), bias2_img)
    c2_img.convolution()
    #c2_img.out[c2_img.out <= 0] = 0

    m2_img = MP(c2_img.out, (2,2), (1,1))
    m2_img.maxpooling()

    # random w koji mi trebaja samo jednom
    if i == 1:
        W1_img = np.random.uniform(size = (m2_img.out.shape[1]*m2_img.out.shape[2], ly1_img))

    f1_img = full_layer(m2_img.out, ly1_img, W1_img)
    #f1_img.out[f1_img.out <= 0] = 0

    if i == 1:
        W2 = np.random.uniform(size = (f1_img.out.shape[1], ly2))

    f2 = full_layer(f1_img.out, ly2, W2)
    #f2.out[f2.out <= 0] = 0

    if i == 1:
        Wo = np.random.uniform(size = (f2.out.shape[1], labels))

    o = output(f2.out, Wo, labels, image_labels)

    Loss_now = o.CE_Loss(o.out, image_labels, o.out.shape[0])

    #print(Wo)
    #===============================================

    """ BACKPROPAGATION """
    d_out, Wo = o.out_backpropagation(f2.out, o.out, image_labels, Wo, lr)

    #print(Wo)

    d_f2, W2 = f2.full_backpropagation(d_out, f2.data, W2, lr)
    d_f1_img, W1_img = f1_img.full_backpropagation(d_f2, f1_img.data, W1_img, lr)
    d_f1_img = d_f1_img.reshape(d_f1_img.shape[0], m2_img.out.shape[1], m2_img.out.shape[2])
    d_m2_img = m2_img.mp_backpropagation(d_f1_img, m2_img.data, (2,2), (1,1))
    d_c2_img, filter2_img, bias2_img = c2_img.convo_backpropagation(d_m2_img, c2_img.data, list(filter2_img), bias2_img, (1,1), lr)
    d_m1_img = m1_img.mp_backpropagation(d_c2_img, m1_img.data, (2,2), (1,1))
    d_c1_img, filter1_img, bias1_img = c1_img.convo_backpropagation(d_m1_img, c1_img.data, list(filter1_img), bias1_img, (9,9), lr)


    Loss.append(Loss_now)
    Loss_iter.append(i)

    if ((i/(learn_iter/10)).is_integer() == True) and (i/(learn_iter/10) != 0.0):
        #print(np.where(o.out[0] == np.max(o.out[0]))[0][0], image_labels[0].index(max(image_labels[0])), i)
        print("{} iteracija od {}".format(i, learn_iter))
        print("Vreme: ", datetime.datetime.now().time())
        print("Loss = {}".format(Loss_now), "\n")


    print(i, '\n')
    i += 1
    print(o.out[:5], '\n')
#===============================================================================

print("Finish: ", datetime.datetime.now().time(), "\n")

plt.plot(Loss_iter, Loss, 'b-')
plt.show()
