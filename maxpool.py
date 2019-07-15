import numpy as np

class MP():

    """ data je iz convo sloja output, filter je tupple (dim1, dim2), stride je isto kao filter """
    def __init__(self, data, filter, stride):

        self.data = data
        self.data_num, self.data_h, self.data_w = data.shape # uzimam dimenzije za kasnije

        self.filter = filter # nzm da li mi treba ovo
        self.filter_h, self.filter_w = filter # isto kao za data

        self.stride_h, self.stride_w = stride

        self.out_h, self.out_w = ((self.data_h - self.filter_h)/self.stride_h + 1,
        (self.data_w - self.filter_w)/self.stride_w + 1) # output dimenzije su (W/H +2P - F_W/H)/S + 1

        if self.out_h.is_integer() == False or self.out_w.is_integer() == False: # ako se ne poklope dimenzije filtera sa inputom, output ne moze da postoji
            raise ValueError("Dimenzije lose, promeni filter dimenzije") # tj. ako deljenje sa stride nije ceo broj onda kita

        self.out_h, self.out_w = int(self.out_h), int(self.out_w) # zbog deljenja postaju float, treba mi int
        self.out = np.zeros((self.data_num, self.out_h, self.out_w))
#===============================================================================

    """ maxpooling je da po filteru iz inputa, vidim koji je max i prebacim samo njega u output """
    def maxpooling(self):

        # biranje slike
        for i in range(self.data_num):

            y1 = out_y = 0
            x2 = self.filter_w # znam da jedino ovde nisam sklonio x,y2 ali se plasim da ih sklonim
            y2 = self.filter_h

            # kretanje po inputu
            while y2 <= self.data_h:

                x1 = out_x = 0

                while x2 <= self.data_w:

                    self.out[i, out_y, out_x] = np.max(self.data[i, y1:y2, x1:x2]) # od odsecka inputa biramo max

                    out_x += 1
                    x1 += self.stride_w
                    x2 += self.stride_w

                out_y += 1
                y1 += self.stride_h
                y2 += self.stride_h


    """ metoda sluzi za vracanje maksimumovog indeksa i koristi se u
    backprop jer je backprop za MP samo slope na maks jer ovi koji su 0 nece biti promenjeni ako ih pomnozim sa slopeom """
    @staticmethod
    def arr_max(arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs


    """ backprop je samo slope iz proslog sloja stavljen na max iz inputa"""
    @staticmethod
    def mp_backpropagation(d_prev_ly, this_layer, filter, stride):

        # neke var koje mi trebaju
        stride_h, stride_w = stride
        filter_h, filter_w = filter
        data_num, layer_h, layer_w = this_layer.shape

        d_layer = np.zeros(this_layer.shape)

        for i in range(data_num):

            # pravljenje coord
            y = layer_y = 0

            # pomeranje po layeru
            while y + filter_h <= layer_h:

                x = layer_x = 0

                while x + filter_w <= layer_w:

                    (z, w) = MP.arr_max(this_layer[i, y:y+filter_h, x:x+filter_w])
                    d_layer[i, y+z, x+w] = d_prev_ly[i, layer_y, layer_x]

                    x += stride_w
                    layer_x += 1
                y += stride_h
                layer_y += 1
                
        return d_layer
