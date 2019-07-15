import numpy as np

"""W = random.uniform(size = (br podatka, duzina vektora podatka, duzina sledeceg sloja))"""
class output():

    def __init__(self, data, W, labels, label_arr):

        self.data = data
        self.W = W
        self.labels_arr = label_arr

        self.out = np.zeros((data.shape[0], labels)) # labels mi obelezava koliko imam out neurona
        for i in range(self.data.shape[0]):
            self.out[i] = self.data[i].reshape(1, self.data[i].shape[0]).dot(self.W[i]) #+ self.bias[i]
        #self.out[self.out <= 0] = 0 # ReLU je legit jedna linija lmao i nema izvod (tj dReLU/dx = 1)
        #self.out = np.exp(self.out)/np.sum(np.exp(self.out))
        self.out = output.sigmoid(self.out)


    @staticmethod
    def sigmoid(data):
        return 1/(1 + np.exp(-1*data))

    @staticmethod
    def d_sigmoid(data):
        return output.sigmoid(data)*(1 - output.sigmoid(data))


    @staticmethod
    def CE_Loss(pred, label):
        return -np.sum(label*np.log(pred))


    @staticmethod
    def out_backpropagation(this_layer, pred, label, W, lr):

        d_W = np.zeros((W.shape[0], W.shape[1], W.shape[2]))
        d_data = np.zeros((this_layer.shape[0], this_layer.shape[1]))
        d_out = np.zeros((pred.shape[0], pred.shape[1]))

        for i in range(this_layer.shape[0]):
            d_out[i] = pred[i] - label[i]
            # verovatno ovde greska, ili kod deklaracije d_W/data
            d_W[i] = d_out[i].reshape(d_out[i].shape[0], 1).dot(this_layer[i].reshape(1, this_layer[i].shape[0])).T # ako padne error vrv je .T jedan od ova dva, vrv drugi
            # izvod za softmax sam uzeo samo softam(1-softmax) bmk
            d_data[i] = W[i].dot(d_out[i].T) * output.d_sigmoid(this_layer[i])
        W += d_W * lr

        return d_data, W
