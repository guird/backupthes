import numpy as np
from ridge import ridge, ridge_corr, bootstrap_ridge
import matplotlib
import gc
import sys
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tables
from scipy.misc import imresize
from scipy.io import loadmat
from scipy import std
from sklearn.linear_model import Ridge as sklridge
import hickle as hkl
from time import time

starttime = time()


TR = 1.0  # seconds
fps = 15
min_delay = 3
#############SELECT SUBJECT, area####################
layer = sys.argv[1]
Subject = sys.argv[2]
ROI = sys.argv[3]#"v1"

###############################################



roilist = [ ROI+'lh', ROI+'rh']#, 'v2lh', 'v2rh']

featuresfolder = "../"

print "made it this far"
"""
Fi = tables.open_file("/vol/ccnlab-scratch1/hugo/vim2/Stimuli.mat")

features_train = Fi.get_node('/st')#Fi.get_node('/train'))

features_test = Fi.get_node('/sv')


train_frames = features_train.shape[0]


frame=0
el=0
featuretrain=np.zeros((train_frames/15,numfeats))
while frame < train_frames:

    chunk = features_train[frame:frame + 15]  # first resize the image
    chunk.transpose((0, 2, 3, 1))
    resizedchunk = np.zeros((15, 3,128, 128))

    for i in range(15):
        resizedchunk[i] = chunk[i]

    featuretrain[el] = np.mean(resizedchunk, axis=0).flatten()

    frame += 15
    el += 1
features_train = 0
train_frames = featuretrain.shape[0]



test_frames = features_test.shape[0]



print "Storing pixel values in feature vector"

frame=0
el=0
featuretest=np.zeros((test_frames/15, numfeats))
while frame < test_frames:
    chunk = features_test[frame:frame + 15]  # first resize the image
    chunk.transpose((0, 2, 3, 1))
    resizedchunk = np.zeros((15, 3, 128, 128))
    for i in range(15):
        resizedchunk[i] = chunk[i]
    featuretest[el] = np.mean(resizedchunk, axis=0).flatten()

    frame += 15
    el += 1
features_test = 0

test_frames = featuretest.shape[0]

print featuretrain.shape
print featuretest.shape
"""

featuretrain = np.float32(hkl.load("/vol/ccnlab-scratch1/hugo/vim2/results/errtrainl"+str(layer)+".hkl"))
featuretest = np.float32(hkl.load("/vol/ccnlab-scratch1/hugo/vim2/results/errtestl"+str(layer)+".hkl"))



"""
featuretrain = np.concatenate((featuretrain,loadmat("/vol/ccnlab-scratch1/hugo/vim2/results/errl"+str(layer)+".mat")['train']), axis=1)#Fi.get_node('/train'))
featuretest = np.concatenate((featuretest, loadmat("/vol/ccnlab-scratch1/hugo/vim2/results/errl"+str(layer)+".mat")['test']), axis=1)
"""
train_frames = featuretrain.shape[0]



test_frames = featuretest.shape[0]
"""
for i in range(train_frames):
    for j in range(featuretrain.shape[1]):
        if np.isnan(featuretrain[i,j]):
            featuretrain[i,j] = 0
        if np.isinf(featuretrain[i,j]):
            featuretrain[i,j] = 1

for i in range(test_frames):
    for j in range(featuretest.shape[1]):
        if np.isnan(featuretest[i,j]):
            featuretest[i,j] = 0
        if np.isinf(featuretest[i,j]):
            featuretest[i,j] = 1
"""
print "Storing pixel values in feature vector"


# choose ROI



print "Loading responses"

respfile = tables.open_file("/vol/ccnlab-scratch1/hugo/vim2/VoxelResponses_subject1.mat")

resptrain = np.float32(np.transpose(respfile.get_node('/rt')[:]))

# resptrain = np.transpose(respfile.get('rt'))

print resptrain.shape

# resptest = np.transpose(respfile.get('rv'))

resptest = np.float32(np.transpose(respfile.get_node('/rv')[:]))

Rresp = []  # training data
Presp = []  # test data

print resptest.shape

for roi in roilist:
    if roi == 'all':
        Rresp = resptrain
        Presp = resptest
        break

    roi_idx = np.nonzero(
        respfile.get_node('/roi/' + roi)[:].flatten() == 1)[0]
    print roi_idx.shape
    if Rresp == []:
        Rresp = resptrain[:, roi_idx]
        Presp = resptest[:, roi_idx]
    else:
        Rresp = np.concatenate((Rresp, resptrain[:, roi_idx]), axis=1)
        Presp = np.concatenate((Presp, resptest[:, roi_idx]), axis=1)

resptest = 0
resptrain = 0

for i in range(Presp.shape[0]):
    for j in range(Presp.shape[1]):
        if np.isnan(Presp[i,j]) or np.isinf(Presp[i,j]):
            Presp[i,j] = 0


for i in range(Rresp.shape[0]):
    for j in range(Rresp.shape[1]):
        if np.isnan(Rresp[i,j]) or np.isinf(Rresp[i,j]):
            Rresp[i,j] = 0


print "Concatenating..."
#RStim=np.roll(featuretrain, -min_delay)[min_delay+2:-(min_delay+2)]
RStim = np.concatenate((np.roll(featuretrain, min_delay, axis=0),
                        np.roll(featuretrain, (min_delay + 1), axis=0),
                        np.roll(featuretrain, (min_delay + 2),axis=0)), axis=1)[min_delay+2:-(min_delay+2)]
featuretrain = 0
#PStim = np.roll(featuretest, -min_delay)[min_delay+2:-(min_delay+2)]
PStim = np.concatenate((np.roll(featuretest, (min_delay), axis=0),
                        np.roll(featuretest, (min_delay + 1),axis=0),
                        np.roll(featuretest, (min_delay + 2),axis=0)), axis=1)[min_delay+2:-(min_delay+2)]
featuretrain = 0
# RStim = zscore(zscore(RStim, axis=1), axis=0)[5:-5]
Rstimmu = np.mean(RStim, axis=0)
Rstimstdev = std(RStim, axis=0)

print PStim.shape
print RStim.shape
gc.collect()
#Remove all 0's and NANs

for i in range(Rstimstdev.shape[0]):
    if Rstimstdev[i] == 0 or np.isnan(Rstimstdev[i]) or np.isinf(Rstimstdev[i]):
        Rstimstdev[i] = 1





RStim = (RStim - Rstimmu)/Rstimstdev



# PStim = zscore(zscore(PStim, axis =1), axis=0)[5:-5]
PStim = (PStim - Rstimmu)/Rstimstdev
# Rresp = zscore(zscore(Rresp, axis=1), axis=0)[5:-5]
Rrespmu = np.mean(Rresp, axis=0)
Rresptdev = std(Rresp, axis=0)
for i in range(Rresptdev.shape[0]):
    if Rresptdev[i] == 0 or np.isnan(Rresptdev[i]) or np.isinf(Rresptdev[i]):
        Rresptdev[i] = 1





Rresp = ((Rresp - Rrespmu)/Rresptdev)[min_delay+2:-(min_delay+2)]
# Presp = zscore(zscore(Presp, axis=1), axis=0)[5:-5]
Presp =((Presp - Rrespmu)/Rresptdev)[min_delay+2:-(min_delay+2)]
alphas = np.logspace(20,0,200)

gc.collect()


print "Shapes:"

print RStim.shape
print PStim.shape
print Rresp.shape
print Presp.shape

print "Starting ridge regression.."

#RStim = 0
#Rresp = 0
"""
top = -10000
corrs =[]
for alph in alphas:
    r = sklridge(alpha =alph, solver='sag')
    
    r.fit(RStim, Rresp)
    print "fitted " + str(alph)
    pred = r.predict(PStim)
    res = pred - Presp
    pred = (1- (np.sum((res)**2, axis=0))/(np.sum(((Presp - Presp.mean(axis=0))**2), axis=0)))
    
    
    numnan = 0
    for c in range(pred.shape[0]):
        if np.isnan(pred[c]) or np.isinf(pred[c]):
            pred[c] = 0
            numnan +=1
    print "nans: " + str(numnan)

            
    print pred.shape
    print np.mean(pred)
    corrs.append(np.mean(pred))
    if np.mean(pred) > top:
        corr = pred
        residuals = res
        top = np.mean(pred)
        print "new best alpha: " + str(alph)
    if time()> starttime + 72000:
        print "Cease now"
        break
if top == -10000:
    corr = pred
"""
corr = ridge_corr(RStim, PStim, Rresp, Presp, alphas)
PStim=0
Presp=0
Rresp =0
RStim=0
gc.collect()
maxalph = np.argmax(np.mean(corr,axis=1))
np.save( "err" + str(layer) + "corr"+ROI+str(Subject)+".npy", corr[maxalph])
print "regdone"
# hkl.dump(corr, "corr"+".hkl")
#savemat('corrs_pix.mat', {'corr': corr})
plt.plot(np.mean(corr, axis=1))
plt.savefig("corrs_alphas.png")
plt.clf()
"""
Remember to plot residuals
"""
#maxalph = np.argmax(np.mean(corr, axis=1))
#print maxalph
#plt.scatter(residuals.flatten(), range(residuals.size))
#plt.savefig("residualsvim.png")
#plt.clf()

plt.hist(corr[maxalph], bins=len(corr[maxalph])/5)
print "ridge complete"
plt.savefig("err" + str(layer) + "corr"+ROI+str(Subject)+".png")
