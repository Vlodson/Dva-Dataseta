import numpy as np


""" data je u 3d ovde (br_podatka, dim1, dim2) i prebacuje se u (br_podatka, dim1*dim2) zbog flattenovanja podataka.
ovi layeri su deo MLP dela mreze. nxt_ly_len je duzina sledeceg sloja (int), W se passuje kao:
random.uniform(size = (br podatka, duzina vektora podatka, duzina sledeceg sloja)) """
class full_layer():

    def __init__(self, data, nxt_ly_len, W):

        # flattenovanje podataka, ne mora uvek
        try: # ima bug za vise full layera pokusava da flattenuje vec flattenovane podatke
            self.data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])

        except IndexError:
            self.data = data # ovo je mali hmm

        self.W = W
        self.out = np.zeros((data.shape[0], nxt_ly_len))
        #---------------------------------------------

        #tehnicki je samo ovo forwardfeed
        for i in range(self.data.shape[0]):
            self.out[i] = self.data[i].reshape(1, self.data[i].shape[0], ).dot(self.W[i]) #+ self.bias[i]
        #self.out[self.out <= 0] = 0 # ReLU je legit jedna linija lmao i nema izvod (tj dReLU/dx = 1)


        self.out = full_layer.sigmoid(self.out)
#===============================================================================

    @staticmethod
    def sigmoid(data):
        return 1/(1 + np.exp(-1*data))

    @staticmethod
    def d_sigmoid(data):
        return full_layer.sigmoid(data)*(1 - full_layer.sigmoid(data))

    """ potrebni su mi slope tezina i podataka. S obzirom da je f(W,podatak) = Suma(W*podatak), po sloju/tezini, izvodi su im W ili podatak puta
    slope sloja pre trenutnog i jos learn_rate"""
    @staticmethod
    def full_backpropagation(d_prev_ly, this_layer, W, lr):

        d_W = np.zeros((this_layer.shape[0], this_layer.shape[1], d_prev_ly.shape[1]))
        d_data = np.zeros((this_layer.shape[0], this_layer.shape[1]))
        #d_bias = np.zeros((this_layer.shape[0], this_layer.shape[1], 1))


        for i in range(this_layer.shape[0]):
            # verovatno ovde greska, ili kod deklaracije d_W/data
            d_W[i] = d_prev_ly[i].reshape(d_prev_ly[i].shape[0], 1).dot(this_layer[i].reshape(1, this_layer[i].shape[0])).T # ako padne error vrv je .T jedan od ova dva, vrv drugi
            d_data[i] = W[i].dot(d_prev_ly[i]) * full_layer.d_sigmoid(this_layer[i])
            #d_bias = isto ko kod convo. receno mi da vidim bez bias

        W = W*0.85 + d_W * lr # gama = 0.85, videti output.py

        return d_data, W
