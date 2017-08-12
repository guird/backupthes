import numpy as np
from ridge import ridge, ridge_corr, bootstrap_ridge
import matplotlib
import sys, gc
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tables
from scipy.misc import imresize
from scipy.stats.mstats import zscore
from skimage.color import rgb2gray as rg
from sklearn.decomposition import PCA

TR = 1.0  # seconds
fps = 15

numfeats =96*96 #128*128*3
min_delay = 3  # times TR

#############SELECT SUBJECT, area####################

Subject = sys.argv[1]
ROI = sys.argv[2]#"v1"

###############################################


print "ROI: " + ROI

roilist = [ ROI+'lh', ROI+'rh']#, 'v2lh', 'v2rh']

features_folder = "../"

Fi = tables.open_file("/vol/ccnlab-scratch1/hugo/vim2/Stimuli.mat")

features_train = (Fi.get_node('/st'))
train_frames = features_train.shape[0]

features_test = (Fi.get_node('/sv'))

test_frames = features_test.shape[0]

print "Storing pixel values in feature vector"


frame = 0
el = 0
featuretrain = np.zeros((train_frames / 15, numfeats))
while frame < train_frames:

    chunk = features_train[frame:frame + 15]  # first resize the image
    chunk.transpose((0, 2, 3, 1))

    #resizedchunk = np.zeros((15, 3,128, 128))
    resizedchunk = np.zeros((15, 96,96))
   
    for i in range(15):
        resizedchunk[i] =  rg(imresize(chunk[i], (96,96)))
   

    featuretrain[el] = np.mean(resizedchunk, axis=0).flatten()

    frame += 15
    el += 1
features_train = 0

print "Resizing all images"

frame = 0
el = 0
featuretest = np.zeros((test_frames / 15, numfeats))
while frame < test_frames:
    chunk = features_test[frame:frame + 15]  # first resize the image
    chunk.transpose((0, 2, 3, 1))
    #resizedchunk = np.zeros((15, 3, 128, 128))
    resizedchunk = np.zeros((15, 96, 96))
   
    for i in range(15):
        resizedchunk[i] = rg(imresize(chunk[i], (96,96)))
    featuretest[el] = np.mean(resizedchunk, axis=0).flatten()
    
    frame += 15
    el += 1
features_test = 0

print featuretrain.shape
print featuretest.shape

"""
pca = PCA(n_components=0.999)
pcad = pca.fit_transform(np.concatenate((featuretrain,featuretest),axis=0))
featuretrain=pcad[:7200]
featuretest=pcad[7200:]
print np.sum(pca.explained_variance_ratio_)
pcad = 0
pca=0
gc.collect()
"""

# choose ROI



print "Loading responses"

respfile = tables.open_file("/vol/ccnlab-scratch1/hugo/vim2/VoxelResponses_subject"+str(Subject)+".mat")

resptrain = np.nan_to_num(np.transpose(respfile.get_node('/rt')[:]))

# resptrain = np.transpose(respfile.get('rt'))

print resptrain.shape

# resptest = np.transpose(respfile.get('rv'))

resptest = np.nan_to_num(np.transpose(respfile.get_node('/rv')[:]))

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

print "Concatenating..."

RStim = np.concatenate((np.roll(featuretrain, -min_delay, axis=0),
                        np.roll(featuretrain, -(min_delay + 1), axis=0),
                        np.roll(featuretrain, -(min_delay + 2),axis=0)), axis=1)[min_delay+2:-(min_delay+2)]
featuretrain = 0

PStim = np.concatenate((np.roll(featuretest, -(min_delay), axis=0),
                        np.roll(featuretest, -(min_delay + 1),axis=0),
                        np.roll(featuretest, -(min_delay + 2),axis=0)), axis=1)[min_delay+2:-(min_delay+2)]
featuretrain = 0
# RStim = zscore(zscore(RStim, axis=1), axis=0)[5:-5]
Rstimmu = np.mean(np.concatenate((RStim, PStim), axis=0), axis=0)
Rstimstdev = np.std(np.concatenate((RStim, PStim), axis=0), axis=0)

for i in range(Rstimstdev.shape[0]):
    if Rstimstdev[i] == 0 or np.isnan(Rstimstdev[i]):
        Rstimstdev[i] = 1

RStim = (RStim - Rstimmu)/Rstimstdev
# PStim = zscore(zscore(PStim, axis =1), axis=0)[5:-5]
PStim = (PStim - Rstimmu)/Rstimstdev
# Rresp = zscore(zscore(Rresp, axis=1), axis=0)[5:-5]
Rrespmu = np.mean(np.concatenate((Rresp, Presp), axis=0), axis=0)
Rresptdev = np.std(np.concatenate((Rresp, Presp), axis=0), axis=0)
for i in range(Rresptdev.shape[0]):
    if Rresptdev[i] == 0 or np.isnan(Rresptdev[i]):
        print i
        Rresptdev[i] = 1
Rresp = ((Rresp - Rrespmu)/Rresptdev)[min_delay+2:-(min_delay+2)]
# Presp = zscore(zscore(Presp, axis=1), axis=0)[5:-5]
Presp =((Presp - Rrespmu)/Rresptdev)[min_delay+2:-(min_delay+2)]
alphas = 10000 *2**np.arange(10)



print "Starting ridge regression.."
corr = ridge_corr(RStim, PStim, Rresp, Presp, alphas)

# hkl.dump(corr, "corr"+".hkl")
#savemat('corrs_pix.mat', {'corr': corr})
plt.plot(np.mean(corr,axis=1))
plt.savefig('corss.png')
plt.clf()
maxalph = np.argmax(np.mean(corr, axis=1))
print maxalph
print np.mean(corr[maxalph])
plt.hist(corr[maxalph], bins=len(corr[maxalph])/5)#(300))
plt.title("Correlation Histogram for Pixels with area:" + ROI + "  Subject " +str(Subject))
plt.xlabel('Correlation')
plt.ylabel('Frequency (voxels)')
np.save("pixelscorr" + ROI+str(Subject)+".np", corr[maxalph])
plt.savefig("pixelscorr"+ROI+str(Subject)+".png")
with open("pixelscorr"+ROI+str(Subject)+".txt", 'w') as t:
    t.write("mean: "+str(np.mean(corr[maxalph])) +", \n max: " 
            + str(corr[maxalph].max()))
    t.close()

