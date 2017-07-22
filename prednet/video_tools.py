import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import numpy as np
from pylab import *



#taken from http://stackoverflow.com/questions/4092927/generating-movie-from-python-without-saving-individual-frames-to-files, modified slightly
dpi = 100

def ani_frame(data, fps, fname, frames):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(data[0,:,:,:])


    def update_img(n ):
        
        im.set_data(data[n,:,:,:])
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img, frames,interval=30)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(fname + '.mp4',writer=writer,dpi=dpi)
    return ani
