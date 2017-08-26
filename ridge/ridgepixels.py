import numpy as np
from ridge import ridge, ridge_corr, bootstrap_ridge
import matplotlib
import sys, gc
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tables
from scipy.misc import imresize
from scipy.stats.mstats import zscore
from skimage.transform import pyramid_reduce
#from skimage.color import rgb2gray as rg
from sklearn.decomposition import PCA
from smooth_and_downsample import smooth_and_downsample

TR = 1.0  # seconds
fps = 15

numfeats =128*128*3
numfeats = 64*64*3
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

n_channels = 3

print "Storing pixel values in feature vector"

###Improving on abstraction
def TR_mean(f):
    #ndarray(seconds,3*len/2*width/2) TR_mean_and_resize(ndarray(frames,3,len,width))
    """
    accepts a video, smooths and downsamples it to halve length and width 
    """
    frame = 0
    el = 0
    frames = f.shape[0]
    numfeats = f.shape[1] * f.shape[2] * f.shape[3]
    seconds = frames/15
    fout = np.zeros((seconds, numfeats))
    while frame <frames:

        chunk = f[frame:frame + 15]  # first resize the image
        chunk.transpose((0, 2, 3, 1))

        resizedchunk = np.zeros((15, 3, f.shape[2], f.shape[3]))
        #resizedchunk = np.zeros((15, 96,96))
   
        for i in range(15):
            resizedchunk[i] = chunk[i]
            # resizedchunk[i] =  pyramid_reduce(
            #     pyramid_reduce(
            #         chunk[i].transpose(0,2,1)).transpose((0,2,1))
            # )
   

        fout[el] = np.mean(resizedchunk, axis=0).flatten()

        frame += 15
        el += 1
    return fout  
###end_func

featuretrain = TR_mean(features_train)

features_train = 0

featuretest = TR_mean(features_test)

features_test = 0

print "Resizing all images"

print featuretrain.shape
print featuretest.shape




pca = PCA(n_components=10000)
pcable = zscore(np.concatenate((featuretrain,featuretest),axis=0))
pcad = pca.fit_transform(pcable)
featuretrain=pcad[:7200]
featuretest=pcad[7200:]
print np.sum(pca.explained_variance_ratio_)
pcad = 0
pca=0
pcable=0
gc.collect()

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

def rollcat(fts, min_delay):
    #ndarray(frames, 3*nfeats) rollcat(ndarray(frames,nfeats)
    """
    copies fts 3 times, rolls them by min_delay, min_delay+1, min_delay+2 respectively, concatenate them. also crops out the first and last 3 entries
    """
    
    return np.concatenate((np.roll(fts, min_delay, axis=0),
                        np.roll(fts, (min_delay + 1), axis=0),
                        np.roll(fts, (min_delay + 2),axis=0)), axis=1)[min_delay+2:-(min_delay+2)]

RStim = rollcat(featuretrain,min_delay)
featuretrain = 0

PStim = rollcat(featuretest, min_delay)

featuretest = 0
#####################ZSCORE DATA####################
# # RStim = zscore(zscore(RStim, axis=1), axis=0)[5:-5]
# Rstimmu = np.mean(np.concatenate((RStim, PStim), axis=0), axis=0)
# Rstimstdev = np.std(np.concatenate((RStim, PStim), axis=0), axis=0)

# for i in range(Rstimstdev.shape[0]):
#     if Rstimstdev[i] == 0 or np.isnan(Rstimstdev[i]):
#         Rstimstdev[i] = 1

# RStim = (RStim - Rstimmu)/Rstimstdev
# # PStim = zscore(zscore(PStim, axis =1), axis=0)[5:-5]
# PStim = (PStim - Rstimmu)/Rstimstdev
# # Rresp = zscore(zscore(Rresp, axis=1), axis=0)[5:-5]
# Rrespmu = np.mean(np.concatenate((Rresp, Presp), axis=0), axis=0)
# Rresptdev = np.std(np.concatenate((Rresp, Presp), axis=0), axis=0)
# for i in range(Rresptdev.shape[0]):
#     if Rresptdev[i] == 0 or np.isnan(Rresptdev[i]):
#         print i
#         Rresptdev[i] = 1
# Rresp = ((Rresp - Rrespmu)/Rresptdev)[min_delay+2:-(min_delay+2)]
# # Presp = zscore(zscore(Presp, axis=1), axis=0)[5:-5]
# Presp =((Presp - Rrespmu)/Rresptdev)[min_delay+2:-(min_delay+2)]

alphas = np.logspace(20,0,200)



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

