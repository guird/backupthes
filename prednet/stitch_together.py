import hickle as hkl
from scipy.io import savemat, loadmat
from scipy.stats.mstats import zscore
import numpy as np
import sys, gc
import tables
from skimage import transform
"""This code takes the extracted error files and stitches them together chronologically and averages them olve 1 second"""
#15 frames
file_overlap = 10
num_test = 15
test_increment = 810
num_train = 200
train_increment = 540
layer = int(sys.argv[1])
#LAYER 1: 122880
#LAYER 2: 614400
#Layer 3: 368640
#layer 4!? 245760

part = np.float32(hkl.load("../vim2/results/layer"+str(layer)+"/trainerr" + str(0) + ".hkl"))
print "layer" + str(layer)
layer_size = part.shape[1]*part.shape[2]*part.shape[3]
#layer_size = layer_size/2
print "layer_size " + str(layer_size)


outtrain= np.zeros((7200, layer_size), dtype=np.float32)#loadmat("../vim2/results/errl1.mat")['train']

def subsample(X):
    # ndarray(seconds, channels, length/2, width/2) 
    # subsample(ndarray(seconds, channels, length, width)
    """
    
    """
    
    seconds = X.shape[0]
    channels = X.shape[1]
    length = X.shape[2]
    width = X.shape[3]
    Xout = np.zeros((seconds,channels,length/2,width/2))
    for i in range(X.shape[0]):
        j = 0
        while j < X.shape[1]:
            
            Xout[i,j:j+3,:,:] = transform.pyramid_reduce(
                 transform.pyramid_reduce(
                     X[i,j:j+3,:,:].transpose(0,2,1)).transpose((0,2,1))
             )

            j+=3
    return Xout

# def test(expr, expec):
#     if expr == expec:
#         print "Success!"
#     else:
#         print expr, "is not equal to", expec

fr = 0

for n in range(num_train):
    
    part = np.float32(hkl.load("../vim2/results/layer"+str(layer)+"/trainerr" + str(n) + ".hkl"))
    
    for i in range(part.shape[0]):
        for j in range(part.shape[1]):
            for k in range(part.shape[2]):
                for l in range(part.shape[3]):
                    if np.isnan(part[i,j,k,l]) or np.isinf(part[i,j,k,l]):
                        part[i,j,k,l] = 0
    print "train filee no. " + str(n)
    copt = part.shape[1]/2
    print copt
    pp =0
    while pp+15 < part.shape[0]:
        
        mini = subsample(part[pp:pp+15])
        
        #mini = mini[:,:copt,:,:] - mini[:,copt:,:,:]
        
        #miniup=mini[:,3:,:,:]#mini[:,:3,:,:]

        

        outtrain[fr] = np.mean(minimini.reshape((15,layer_size)), axis=0)
        fr +=1
        pp +=15

    mini = 0 
    gc.collect()


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
    copt = part.shape[1]/2
    while pp+15 < part.shape[0]:
        
        mini = (part[pp:pp + 15])
        #mini = mini[:,:copt,:,:] - mini[:,copt:,:,:]

        
        outtest[fr] = np.mean(mini.reshape((15, layer_size)), axis=0)




        fr += 1
        pp += 15

    gc.collect()
print outtrain.nbytes      

#outtest = zscore(outtest,axis=0)

hkl.dump(outtrain,"../vim2/results/serrtrainl"+str(layer)+".hkl")
hkl.dump(outtest, "../vim2/results/serrtestl"+str(layer)+".hkl")
#savemat("../vim2/results/errl"+ str(layer)+ ".mat", {"train":outtrain, "test":outtest})
