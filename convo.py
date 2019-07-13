from img_pp import images, image_labels
#from snd_pp import sounds, sound_labels
import numpy as np

data_im = images[:10]
#data_sd = sounds[:10]

class convo():

    """ data mora biti u obliku (br_podatka, 1. dim, 2. dim); filter se pise u obliku (br_filtera, 1. dim, 2.dim),
    stride je za koliko ce se pomerati filter i dolazi u obliku vertikalni pomeraj,
    i horizontalni pomeraj """
    def __init__(self, data, filter, stride):

        self.data = data
        self.data_num, self.data_h, self.data_w = data.shape

        # filteri u cnn su kao weightovi u obicnoj nn
        # IDE SAMO JEDAN FILTER PO KLASI
        self.filter_num, self.filter_h, self.filter_w = filter # ovde nema shape jer je filter tupple
        self.filter = []
        for i in range(self.filter_num):
            self.filter.append(np.random.randn(self.filter_h, self.filter_w))
        self.filter = np.array(self.filter)

        self.bias = np.zeros((self.filter_num, 1))

        self.stride_h, self.stride_w = stride # stride po width i po height
        self.out_h, self.out_w = ((self.data_h - self.filter_h)/self.stride_h + 1,
        (self.data_w - self.filter_w)/self.stride_w + 1) # output dimenzije su (W/H +2P - F_W/H)/S + 1

        if self.out_h.is_integer() == False or self.out_w.is_integer() == False: # ako se ne poklope dimenzije filtera sa inputom, output ne moze da postoji
            raise ValueError("Dimenzije lose, promeni filter dimenzije") # tj. ako deljenje sa stride nije ceo broj onda kita

        self.out_h, self.out_w = int(self.out_h), int(self.out_w) # zbog deljenja postaju float, treba mi int
        self.out = np.zeros((self.data_num, self.out_h, self.out_w)) # popunjavam sa nulama, da mogu kasnije da zamenim sa normalnim vrednostima

#===============================================================================

    # konvolucija je u sustini prelazenje filtera preko nekog dela podatka i onda gledanje koliko se to poklapa
    def convolution(self):
        for i in range(self.data_num): # za svaki podatak

            # spremanje coord za kretanje filtera po podatku
            x1 = y1 = 0
            x2 = self.filter_w
            y2 = self.filter_h
            out_x = out_y = 0

            for j in range(self.filter_num): # za svaki filter
                while y2 <= self.data_h: # dokle god nisi presao visinu podatka
                    while x2 <= self.data_w: # dokle god nisi presao sirinu podatka

                        # output konv sloja je suma element wise producta filtera i tog dela podatka
                        self.out[i][out_y][out_x] = np.sum(self.filter[j]*self.data[i, y1:y2, x1:x2]) + self.bias[j] # mozda je j kod self.out
                        # self.filter[j].dot(self.data[i, y1:y2, x1:x2]) dot product (ne radi)
                        x1 += self.stride_w # samo povecavam za korak obe coord
                        x2 += self.stride_w
                        out_x += 1 # povecavam mesto u outputu

                    y1 += self.stride_h
                    y2 += self.stride_h
                    out_y += 1
#===============================================================================

    @staticmethod
    def backpropagation(d_prev_ly, this_ly, filter, stride):

        # neke var koje mi trebaju
        filter_num, filter_h, filter_w = filter.shape
        data_num, layer_h, layer_w = this_layer.shape

        # pravim slope varijable i punim ih nulama
        d_layer = np.zeros(this_layer.shape)
        d_filter = np.zeros(filter.shape)
        d_bias = np.zeros((filter_num, 1))

        # za svaku sliku
        for i in range(data_num):
            # za svaki filter
            for j in range(n_filter):

                # pravljenje coord
                x1 = y1 = 0
                x2 = filter_w
                y2 = filter_h
                layer_x = layer_y = 0

                # pomeranje po layeru
                while y2 <= layer_h:
                    while x2 <= layer_w:

                        # slope filtera je slope outputa conv na coord outputa * inputov deo podatka kojem odgovara filter
                        d_filter[j] += this_layer[i, y1:y2, x1:x2] * d_prev_ly[i, layer_y, layer_x]
                        # slope dela inputa je filter * sa outputom koji mu odgovara
                        d_layer[i, y1:y2, x1:x2] += filter[j] * d_prev_ly[i, layer_y, layer_x]

                        # nastavljanje kretanja
                        x1 += s
                        x2 += s
                        layer_x += 1
                    y1 += s
                    y2 += s
                    layer_y += 1

                # bias je samo suma slopa outputa za taj layer
                d_bias[j] = np.sum(d_prev_ly[i])

        return d_this_layer, d_filter, d_bias
