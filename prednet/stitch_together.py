import hickle as hkl
from scipy.io import savemat, loadmat
import numpy as np
import sys
"""This code takes the extracted error files and stitches them together chronologically and averages them olve 1 second"""
#15 frames
file_overlap = 10
num_test = 15
test_increment = 810
num_train = 200
train_increment = 540
layer = int(sys.argv[1])
print layer
if layer == 0:
    layer_size = 122880
elif layer==1:
    layer_size = 614400
elif layer==2:
    layer_size = 368640
elif layer==3:
    layer_size = 122880

#LAYER 1: 122880
#LAYER 2: 614400
#Layer 3: 368640
#layer 4!? 245760

outtrain= np.zeros((7200, layer_size), dtype=np.float32)#loadmat("../vim2/results/errl1.mat")['train']



fr = 0

for n in range(num_train):
    part = hkl.load("../vim2/results/layer"+str(layer)+"/trainerr" + str(n) + ".hkl")
    print part.shape
    for i in range(part.shape[0]):
        for j in range(part.shape[1]):
            for k in range(part.shape[2]):
                for l in range(part.shape[3]):
                    if np.isnan(part[i,j,k,l]) or np.isinf(part[i,j,k,l]):
                        part[i,j,k,l] = 0
    print "train filee no. " + str(n)
    print part.shape
    pp =0
    while pp+15 < part.shape[0]:
        mini = part[pp:pp+15]
        print mini.shape

        #miniup=mini[:,3:,:,:]#mini[:,:3,:,:]

        print mini.shape

        outtrain[fr] = np.mean(mini.reshape((15,layer_size)), axis=0)
        fr +=1
        pp +=15




outtest = np.zeros((540, layer_size))

fr = 0
for n in range(num_test):
    part = hkl.load("../vim2/results/layer"+str(layer)+"/testerr" + str(n) + ".hkl")#[1:,:layer_size]

    for i in range(part.shape[0]):
        for j in range(part.shape[1]):
            for k in range(part.shape[2]):
                for l in range(part.shape[3]):
                    if np.isnan(part[i,j,k,l]) or  np.isinf(part[i,j,k,l]):
                        part[i,j,k,l] = 0
                        print "nan replaced"
    print "test file no. " + str(n)
    pp = 0
    print part.shape


    while pp+15 < part.shape[0]:
        print pp

        mini = part[pp:pp + 15]
        print mini.shape
        #mini = mini[:, :3, :, :] + mini[:, 3:, :, :]
        print mini.shape
        outtest[fr] = np.mean(mini.reshape((15, layer_size)), axis=0)




        fr += 1
        pp += 15

    print pp
    print fr

#hkl.dump(outtrain,"../vim2/results/errtrainl1.hkl")
savemat("../vim2/results/errl"+ str(layer)+ ".mat", {"train":outtrain, "test":outtest})
