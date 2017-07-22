
import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py as h
import hickle as hkl
import theano
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
import time
import sys
from prednet import PredNet
#from data_utils import SequenceGenerator

starttime = time.time()

WEIGHTS_DIR = "model_data"
DATA_DIR = "../vim2/preprocessed"
RESULTS_SAVE_DIR = "../vim2/results/"

n_plot = 40
batch_size = 15
fperbat = 810

nt = 10 #fperbat + 10
tot_frames = 8100
nbat = tot_frames/fperbat

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = DATA_DIR+ '/vim2_test'
#test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'all_error'
dim_ordering = layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(input=inputs, output=predictions)

X_test = np.zeros((1,10,3,128,160))
X_test[0,:,:,:,:] = (hkl.load('../vim2/preprocessed/test.hkl')[0:10]).transpose((0,3,1,2))

e = test_model.predict(X_test)

print e.shape