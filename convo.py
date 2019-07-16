import numpy as np

class convo():

    """ data mora biti u obliku (br_podatka, 1. dim, 2. dim); filter se pise u obliku (br_filtera, 1. dim, 2.dim),
    stride je za koliko ce se pomerati filter i dolazi u obliku vertikalni pomeraj,
    i horizontalni pomeraj. bias se pokrece kao np.zeros((1,1)),
    filter se pokrece kao (filter_h, filter_w,np.random.uniform(size=(filter_h, filter_w)))"""
    def __init__(self, data, filter, stride, bias):

        self.data = data
        self.data_num, self.data_h, self.data_w = data.shape

        # filteri u cnn su kao weightovi u obicnoj nn
        self.filter_h, self.filter_w, self.filter = filter # ovde nema shape jer je filter tupple

        self.bias = bias # stae bias

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
            y = out_y = 0

            while y + self.filter_h <= self.data_h: # dokle god nisi presao visinu podatka

                x = out_x = 0

                while x + self.filter_w <= self.data_w: # dokle god nisi presao sirinu podatka

                    # output konv sloja je suma element wise producta filtera i tog dela podatka
                    self.out[i, out_y, out_x] = np.sum(self.filter*self.data[i, y:y + self.filter_h, x:x + self.filter_w]) + self.bias
                    x += self.stride_w # samo povecavam za korak obe coord
                    out_x += 1 # povecavam mesto u outputu

                y += self.stride_h
                out_y += 1
#===============================================================================

    """ U sustini izvod za convo je isti kao i za obican MLP, za filter je input
    puta slope sa proslog sloja s tim sto je sada input neki 2d deo podatka, ne samo
    1d broj koji obelezava podatak. sto se tice slopea inputa, to je i dalje
    filter * slope sa proslog sloja s tim sto opet to je slope za 2d deo podatka"""
    @staticmethod
    def convo_backpropagation(d_prev_ly, this_layer, filter, bias, stride, lr):

        # neke var koje mi trebaju
        stride_h, stride_w = stride
        filter_h, filter_w = filter[0], filter[1] # ovde je bio filter.shape
        data_num, layer_h, layer_w = this_layer.shape

        # pravim slope varijable i punim ih nulama
        d_layer = np.zeros(this_layer.shape)
        d_filter = np.zeros((filter[0], filter[1]))
        d_bias = np.zeros((1, 1)) # !!

        # za svaku sliku
        for i in range(data_num):

            # pravljenje coord
            y = layer_y = 0

            # pomeranje po layeru
            while y + filter_h <= layer_h:

                x = layer_x = 0

                while x + filter_w <= layer_w:

                    # slope filtera je slope outputa conv na coord outputa * inputov deo podatka kojem odgovara filter
                    # MOZDA OVAJ += MOZE DA SJEBE MREZU, NISAM SIGURAN
                    d_filter += this_layer[i, y:y + filter_h, x:x + filter_w] * d_prev_ly[i, layer_y, layer_x]
                    # slope dela inputa je filter * sa outputom koji mu odgovara
                    d_layer[i, y:y + filter_h, x:x + filter_w] += filter[2] * d_prev_ly[i, layer_y, layer_x]

                    # nastavljanje kretanja
                    x += stride_w
                    layer_x += 1
                y += stride_h
                layer_y += 1

            # bias je samo suma slopa outputa za taj layer
            d_bias = np.sum(d_prev_ly[i])

        filter[2] = filter[2]*0.85 + d_filter * lr # gama = 0.85, videti output.py
        bias += d_bias * lr


        return d_layer, filter, bias
