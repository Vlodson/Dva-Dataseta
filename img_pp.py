""" Ova skripta sluzi za ucitavanje, preprocesiranje slika, pravljenje labela i
pravljenje trening seta sa nasumicno rasporedjenim trening slikama + iz trening
slika pravljenje seta za trening testiranja i pravljenje test seta
1 je za cello, 2 je flute, 3 je sax, 4 je violin"""

import numpy as np
from PIL import Image
import os


# preko os.listdir uzima sve u folderu i appenduje ih na listu instrumenta tako
# uzimajuci sve slike iz foldera. Sa asarray delom pretvaram sliku odma u np.arr
# jos jednom labele: 1 - cello   2 - flute   3 - sax   4 - violin
def ucitavanje(instrument, path, br_labele):
    inst = []
    inst_label = []

    for i in os.listdir("Img_data/{}/{}".format(path, instrument)):
        inst.append(np.asarray(Image.open("Img_data/{}/{}/{}".format(path, instrument, i))))
        inst_label.append("{}".format(br_labele))

    return inst, inst_label


cello, cello_label = ucitavanje("cello", "Training", 1)
flute, flute_label = ucitavanje("flute", "Training", 2)
sax, flute_label = ucitavanje("sax", "Training", 3)
violin, flute_label = ucitavanje("violin", "Training", 4)


# array mora da bude
def color_filter(rmin, rmax, gmin, gmax, bmin, bmax, array):
    for i in range(np.shape(array)[0]):
        array[i].setflags(write = 1)
        for j in range(np.shape(array)[1]):
            for k in range(np.shape(array)[2]):
                if array[i][j][k][0] >= rmin/2 or array[i][j][k][0] <= rmax/2:
                    array[i][j][k][0] = 0
                if array[i][j][k][1] >= gmin/2 or array[i][j][k][1] <= gmax/2:
                    array[i][j][k][1] = 0
                if array[i][j][k][2] >= bmin/2 or array[i][j][k][2] <= bmax/2:
                    array[i][j][k][2] = 0

    return array

cello = color_filter(90, 255, 73, 217, 50, 165, cello)
flute = color_filter(117, 255, 107, 255, 95, 255, flute)
sax = color_filter(90, 255, 90, 255, 0, 6, sax)
violin = color_filter(90, 255, 73, 217, 50, 165, violin)
