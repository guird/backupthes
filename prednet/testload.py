'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Dense, Flatten
import h5py
import json

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *


n_plot = 40
batch_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')



# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
conf = json.loads(json_string)

train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
wf = h5py.File(weights_file, mode='r')
train_model.load_weights(weights_file)
