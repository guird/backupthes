import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROI = "v1"
corr1=np.load("pixelscorr"+ROI+"1"+".np.npy")
corr2=np.load("pixelscorr"+ROI+"2"+".np.npy")
corr3=np.load("pixelscorr"+ROI+"3"+".np.npy")

plt.plot(corr1, 'b')
plt.plot(corr1, 'r')
plt.plot(corr1, 'g')

plt.savefig(ROI+"allcorrs.png")


    
