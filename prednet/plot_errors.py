import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import hickle as hkl

dat = hkl.load("../vim2/results/error_outputs")
fig = plt.figure()
plt.imshow(dat)

fig.savefig("errors.png")
