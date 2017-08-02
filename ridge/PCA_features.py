import numpy as np
import sys
from scipy.io import loadmat,savemat
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
import tables
layer = sys.argv[1]

data = loadmat("../vim2/results/errl"+str(layer)+".mat")["train"]
"""
Fi = tables.open_file("/vol/ccnlab-scratch1/hugo/vim2/Stimuli.mat")

dat = np.array(Fi.get_node('/sv'))
data = np.zeros((dat.shape[0],dat.shape[1]*dat.shape[2]*dat.shape[3]))
for i in range(data.shape[0]):
    data[i]=dat[i,:,:,:].flatten()
Fi.close()
print data.shape
"""
#data= np.concatenate((data,loadmat("../vim2/results/errl"+str(layer)+".mat")["test"]), axis=0)
print data.shape
print np.amax(data)
print np.amin(data)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if np.isnan(data[i,j]) or np.isinf(data[i,j]):
            data[i,j] = 0 

print data.shape
comps = 10000
ipca = IncrementalPCA(n_components=10000)
trans = ipca.fit_transform(data)
print ipca.explained_variance_ratio_
print np.sum(ipca.explained_variance_ratio_)
#print np.sum(np.var(data))) / np.var(ipca, axis=0) 
print trans.shape
savemat("PCAl"+str(layer)+".mat", {"pca":trans})
