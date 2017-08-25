import numpy as np
import sys, gc
#from scipy.io import loadmat,savemat
from sklearn.decomposition import IncrementalPCA 
#import PCA, IncrementalPCA, KernelPCA
#import tables
import hickle as hkl
layer = sys.argv[1]

data = hkl.load("../vim2/results/errtrainl"+str(layer)+".hkl")#["train"]
#data = loadmat("../vim2/results/errl"+str(layer)+".mat")["train"]

"""
Fi = tables.open_file("/vol/ccnlab-scratch1/hugo/vim2/Stimuli.mat")

dat = np.array(Fi.get_node('/sv'))
data = np.zeros((dat.shape[0],dat.shape[1]*dat.shape[2]*dat.shape[3]))
for i in range(data.shape[0]):
    data[i]=dat[i,:,:,:].flatten()
Fi.close()
print data.shape
"""
print data.nbytes
data= np.concatenate((data,hkl.load("../vim2/results/errtestl"+str(layer)+".hkl")),axis=0)

# data= np.concatenate((data,loadmat("../vim2/results/errl"+str(layer)+".mat")["test"]), axis=0)
print data.shape
print np.amax(data)
print np.amin(data)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if np.isnan(data[i,j]) or np.isinf(data[i,j]):
            data[i,j] = 0 

print data.shape

ipca = IncrementalPCA(n_components=None)
trans = ipca.fit_transform(data)

print np.sum(ipca.explained_variance_ratio_)
"""
while np.sum(ipca.explained_variance_ratio_)<desired_evr:
    comps += 1000
    ipca=0
    trans=0
    gc.collect()
    ipca = IncrementalPCA(n_components=comps)
    trans = ipca.fit_transform(data)
   """ 
    

#print np.sum(np.var(data))) / np.var(ipca, axis=0) 
print trans.shape
hkl.dump(trans[:7200],"PCAtrainl"+str(layer)+".hkl") 

hkl.dump(trans[7200:],"PCAtestl"+str(layer)+".hkl") 

# savemat("PCAl"+str(layer)+".mat", {"pca":trans})
