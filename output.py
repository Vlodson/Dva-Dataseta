import numpy as np

"""W = random.uniform(size = (br podatka, duzina vektora podatka, duzina sledeceg sloja))"""
class output():

    def __init__(self, data, W, labels, label_arr):

        self.data = data
        self.W = W
        self.labels_arr = label_arr

        self.out = np.zeros((data.shape[0], labels)) # labels mi obelezava koliko imam out neurona
        for i in range(self.data.shape[0]):
            self.out[i] = self.data[i].dot(self.W) #+ self.bias[i]
        #self.out = self.out/np.mean(self.out)
        #self.out[self.out <= 0] = 0 # ReLU je legit jedna linija lmao i nema izvod (tj dReLU/dx = 1)
        #self.out = np.exp(self.out)/np.sum(np.exp(self.out))
        self.out = output.softmax(self.out)


    @staticmethod
    def softmax(data):
        temp = []
        for i in range(data.shape[0]):
            temp.append(np.exp(data[i]) / np.exp(data[i]).sum())
        temp = np.array(temp)
        return temp

    @staticmethod
    def d_softmax(data):
        return output.softmax(data)*(1 - output.softmax(data))


    @staticmethod
    def CE_Loss(pred, label, num_data):
        return -np.sum(label*np.log(pred))


    @staticmethod
    def out_backpropagation(this_layer, pred, label, W, lr):

        d_W = np.zeros((W.shape[0], W.shape[1]))
        d_data = np.zeros((this_layer.shape[0], this_layer.shape[1]))
        d_out = np.zeros((pred.shape[0], pred.shape[1]))

        d_out = pred - label
        # verovatno ovde greska, ili kod deklaracije d_W/data
        d_W = this_layer.T.dot(d_out) # ako padne error vrv je .T jedan od ova dva, vrv drugi
        # izvod za softmax sam uzeo samo softmax(1-softmax) bmk
        d_data = d_out.dot(W.T) * output.d_softmax(this_layer)

        #print(d_W[:5])
        #W = W*0.85 + d_W * lr # MOMENTUM TI JE 0.55 VIDETI SAJT: http://ruder.io/optimizing-gradient-descent/index.html#momentum
        W -= d_W*lr
        #W = W*0.95

        return d_data, W
