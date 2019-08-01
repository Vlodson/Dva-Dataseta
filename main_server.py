import numpy as np
from convo import convo
from maxpool import MP
from full import full_layer
from output import output
from img_pp import images, image_labels
print("slike ucitane")
from snd_pp import sounds, sound_labels
print("zvuk ucitan")
import datetime
import matplotlib.pyplot as plt
#===============================================================================

# file writing
def write_data(file_name, data):

    with open(file_name, 'ab') as fn:

        np.savetxt(fn, data, delimiter = ',')
#===============================================================================


# randomizuje redosled podataka
for i in range(len(sounds)):
    rand = np.random.randint(0, len(sounds))

    temp_i = images[i]
    images[i] = images[rand]
    images[rand] = temp_i

    temp_i = image_labels[i]
    image_labels[i] = image_labels[rand]
    image_labels[rand] = temp_i
    #---

    temp_s = sounds[i]
    sounds[i] = sounds[rand]
    sounds[rand] = temp_s

    temp_s = sound_labels[i]
    sound_labels[i] = sound_labels[rand]
    sound_labels[rand] = temp_s
#===============================================================================
num_data = images.shape[0] # upises obicno broj, ali ako hoces sve, onda data.shape[0]

images = images[:num_data]
image_labels = image_labels[:num_data]

sounds = sounds[:num_data]
sound_labels = sound_labels[:num_data] # nzm da li treba ovo uopste
#===============================================================================

labels = 4 # violina, sax, violoncelo, flauta
lr = 0.000000001 # learn rate, treba da bude mali za pomeranje po slopeovima
learn_iter = 100 # za broj iteracija ucenja
i = 1

# prve random var
filter1_img = (15, 15, np.random.randn(15,15))
filter1_snd = (10, 22, np.random.randn(10,22))
bias1_img = np.zeros((1,1))
bias1_snd = np.zeros((1,1))
#---
filter2_img = (3, 3, np.random.randn(3, 3))
filter2_snd = (4, 8, np.random.randn(4, 8))
bias2_img = np.zeros((1,1))
bias2_snd = np.zeros((1,1))

# neke const
ly1_img = 8
ly1_snd = 8

ly2 = 10

Loss = []
Loss_prev = 1000000 # posto gledam da li je Loss_now manji od Loss_prev,
Loss_now = 1000000 #  ako ih stavim na 0 na pocetku, bice Loss_prev uvek manji
Loss_iter = []
# ly2_img = 6
# ly2_snd = 6

labels_img = np.shape(image_labels)[1]
labels_snd = np.shape(sound_labels)[1]
#===============================================================================

print("Start: ", datetime.datetime.now().time())

while i <= learn_iter:

    print(i)
    # za ispis podataka dole
    Loss_prev = Loss_now

    """ FEEDFORWARD """
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

    # random w koji mi trebaja samo jednom
    if i == 1:
        W1_img = np.random.randn(images.shape[0], m2_img.out.shape[1]*m2_img.out.shape[2], ly1_img)
        W1_snd = np.random.randn(sounds.shape[0], m2_snd.out.shape[1]*m2_snd.out.shape[2], ly1_snd)
    #---

    f1_img = full_layer(m2_img.out, ly1_img, W1_img) # ovde se vec radi forwardfeed, nema f za to
    f1_img.out[f1_img.out <= 0] = 0
    f1_snd = full_layer(m2_snd.out, ly1_snd, W1_snd) # ovde se vec radi forwardfeed, nema f za to
    #f1_snd.out[f1_snd.out <= 0] = 0
    #---

    f1_out = np.zeros((images.shape[0], f1_img.out.shape[1] + f1_snd.out.shape[1])) # konkatiniran output proslog sloja

    for j in range(images.shape[0]):
        f1_out[j] = np.append(f1_img.out[j], f1_snd.out[j])
    #---

    if i == 1:
        W2 = np.random.randn(images.shape[0], f1_out.shape[1], ly2)
    #---

    f2 = full_layer(f1_out, ly2, W2)
    #f2.out[f2.out <= 0] = 0
    #---

    if i == 1:
        Wo = np.random.randn(images.shape[0], f2.out.shape[1], labels)
    #---

    o = output(f2.out, Wo, labels, image_labels)
    o.out[o.out <= 0] = 0
    #===============================================

    """ BACKPROPAGATION """
    d_out, Wo = o.out_backpropagation(f2.out, o.out, image_labels, Wo, lr)
    #---

    d_f2_img, W2_img = f2.full_backpropagation(d_out, f2.data[:, 0:f1_img.out.shape[1]], W2[:, :f1_img.out.shape[1], :], lr)
    d_f2_snd, W2_snd = f2.full_backpropagation(d_out, f2.data[:, f1_img.out.shape[1]:f1_img.out.shape[1] + f1_snd.out.shape[1]], W2[:, f1_img.out.shape[1]:f1_img.out.shape[1] + f1_snd.out.shape[1], :], lr)
    W2 = np.append(W2_img, W2_snd)
    W2 = W2.reshape(images.shape[0], f1_img.out.shape[1]+f1_snd.out.shape[1], ly2)

    # d_f2_img = d_f2[: ,0:f1_img.out.shape[1]]
    # d_f2_snd = d_f2[: ,f1_img.out.shape[1]:f1_snd.out.shape[1]]
    #---

    d_f1_img, W1_img = f1_img.full_backpropagation(d_f2_img, f1_img.data, W1_img, lr)
    d_f1_snd, W1_snd = f1_snd.full_backpropagation(d_f2_snd, f1_snd.data, W1_snd, lr)
    #---

    d_f1_img = d_f1_img.reshape(d_f1_img.shape[0], m2_img.out.shape[1], m2_img.out.shape[2])
    d_f1_snd = d_f1_snd.reshape(d_f1_snd.shape[0], m2_snd.out.shape[1], m2_snd.out.shape[2])

    d_m2_img = m2_img.mp_backpropagation(d_f1_img, m2_img.data, (2,2), (1,1))
    d_m2_snd = m2_snd.mp_backpropagation(d_f1_snd, m2_snd.data, (3,3), (2,2))
    #---

    d_c2_img, filter2_img, bias2_img = c2_img.convo_backpropagation(d_m2_img, c2_img.data, list(filter2_img), bias2_img, (1,1), lr) # filter mora u list jer tupple ne da da se menjaju stvari
    d_c2_snd, filter2_snd, bias2_snd = c2_snd.convo_backpropagation(d_m2_snd, c2_snd.data, list(filter2_snd), bias2_snd, (2,2), lr)
    #---

    d_m1_img = m1_img.mp_backpropagation(d_c2_img, m1_img.data, (2,2), (1,1))
    d_m1_snd = m1_snd.mp_backpropagation(d_c2_snd, m1_snd.data, (1,2), (1,2))
    #---

    d_c1_img, filter1_img, bias1_img = c1_img.convo_backpropagation(d_m1_img, c1_img.data, list(filter1_img), bias1_img, (9,9), lr) # filter u listu ne tupple
    d_c1_snd, filter1_snd, bias1_snd = c1_snd.convo_backpropagation(d_m1_snd, c1_snd.data, list(filter1_snd), bias1_snd, (8,8), lr)
    #===============================================

    Loss_now = o.CE_Loss(o.out, image_labels)
    #---

    # za cuvanje najboljih tezina, tj. tamo gde je loss bio najmanji
    if Loss_now < Loss_prev:
        filter1_img_best = filter1_img
        filter2_img_best = filter2_img

        bias1_img_best = bias1_img
        bias2_img_best = bias2_img

        W1_img_best = W1_img
        W2_img_best = W2_img
        #---

        filter1_snd_best = filter1_snd
        filter2_snd_best = filter2_snd

        bias1_snd_best = bias1_snd
        bias2_snd_best = bias2_snd

        W1_snd_best = W1_snd
        W2_snd_best = W2_snd

        Wo_best = Wo
    #---

    Loss.append(Loss_now)
    Loss_iter.append(i)

    if ((i/(learn_iter/10)).is_integer() == True) and (i/(learn_iter/10) != 0.0):
        #print(np.where(o.out[0] == np.max(o.out[0]))[0][0], image_labels[0].index(max(image_labels[0])), i)
        print("{} iteracija od {}".format(i, learn_iter))
        print("Vreme: ", datetime.datetime.now().time())
        print("Loss = {}".format(Loss_now), "\n")

    i += 1
#===============================================================================
print("Finish: ", datetime.datetime.now().time(), "\n")

# za server je ovo iskomentarisano
# plt.plot(Loss_iter, Loss, 'b-')
# plt.show()

write_data("W_b/Image/filter1_img.csv", filter1_img_best[2])
write_data("W_b/Image/filter2_img.csv", filter2_img_best[2])
write_data("W_b/Image/bias1_img.csv", bias1_img_best)
write_data("W_b/Image/bias2_img.csv", bias2_img_best)
write_data("W_b/Image/W1_img.csv", W1_img_best.reshape(W1_img_best.shape[0], W1_img_best.shape[1]*W1_img_best.shape[2]))
write_data("W_b/Image/W2_img.csv", W2_img_best.reshape(W2_img_best.shape[0], W2_img_best.shape[1]*W2_img_best.shape[2]))

write_data("W_b/Sound/filter1_snd.csv", filter1_snd_best[2])
write_data("W_b/Sound/filter2_snd.csv", filter2_snd_best[2])
write_data("W_b/Sound/bias1_snd.csv", bias1_snd_best)
write_data("W_b/Sound/bias2_snd.csv", bias2_snd_best)
write_data("W_b/Sound/W1_snd.csv", W1_snd_best.reshape(W1_snd_best.shape[0], W1_snd_best.shape[1]*W1_snd_best.shape[2]))
write_data("W_b/Sound/W2_snd.csv", W2_snd_best.reshape(W2_snd_best.shape[0], W2_snd_best.shape[1]*W2_snd_best.shape[2]))

write_data("W_b/Wo.csv", Wo_best.reshape(Wo_best.shape[0], Wo_best.shape[1]*Wo_best.shape[2]))
