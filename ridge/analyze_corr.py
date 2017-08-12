import numpy as np
import scipy.stats as stats
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys 
import cortex
#https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
def factors(n):    
    return  list( ((i, n//i) for i in range(1, int(n**0.5) + 1) if n % i == 0))

ROI = sys.argv[1]
layer = sys.argv[2]
subject = sys.argv[3]

#pcorr1=np.load("pixelscorr"+ROI+subject+".np.npy")
#pcorr2=np.load("pixelscorr"+ROI+"2"+".np.npy")
#pcorr3=np.load("pixelscorr"+ROI+"3"+".np.npy")


df = 540-1
critical_p_value=0.01/pcorr1.shape[0]
critical_t_value = stats.t.isf(critical_p_value,df)
critical_r_value = np.sqrt(
    1 / (1 + df/critical_t_value**2 ))
print critical_r_value


zerosurface = (64


#f0corr1=np.load("err"+layer+"corr"+ROI+subject+".npy")

#f0corr2=np.load("err"+layer+"corr"+ROI+"2"+".np.npy")

#f0corr3=np.load("err"+layer+"corr"+ROI+"3"+".np.npy")
"""
fig = plt.figure()
ax = matplotlib.axis.Axis(fig)
thp = plt.fill(pcorr1,'g')
thf0 = plt.fill(f0corr1,'b')

i = 0
for pat in thp:
    pat.zorder = pcorr1[i]
    i+=1
i = 0
for pat in thf0:
    pat.zorder=f0corr1[i]
    i +=1

plt.savefig(ROI+layer+subject+"plt.png")

plt.clf()

plt.hist(f0corr1-pcorr1, bins=250)

plt.savefig(ROI+layer+subject+"hist.png")
"""
