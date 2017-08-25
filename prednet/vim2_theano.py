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

import sys
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
import gc

from prednet import PredNet

from time import time
from scipy.io import loadmat
from video_tools import ani_frame
starttime = time()


for g in (0,):
    
    WEIGHTS_DIR = "model_data"
    DATA_DIR = "../vim2/preprocessed/"
    RESULTS_SAVE_DIR = "../vim2/results"
    WEIGHTS_OUT_DIR = "vim2_weights"
    

    n_plot = 40
    sample_size = 10


    weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    training_file = os.path.join(DATA_DIR, 'train')
    out_file = os.path.join(WEIGHTS_OUT_DIR, "vim2_weights")
    weights_file = out_file
    json_file = out_file + ".json"
    
    num_epochs = 10
    num_samples = 108000
    batch_size = 5400
    num_batches = int(num_samples/batch_size)
    nt = 15
    #load model
    
    f = open(json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    train_model.load_weights(weights_file)
    
    #train from scratch
    
    print "initializing"
    input_shape = (3, 128, 160)
    stack_sizes = (input_shape[0], 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    layer_loss_weights = np.array([1., 0., 0., 0.])
    layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
    time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))
    time_loss_weights[0] = 0


    prednet = PredNet(stack_sizes, R_stack_sizes,
                      A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode='error', return_sequences=True, dim_ordering='th', weights=train_model.layers[1].get_weights())
    
    inputs = Input(shape=(nt,) + input_shape)
    errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)

    model = Model(input=inputs, output=errors)
    model.compile(loss='mean_absolute_error', optimizer='adam')
    train_model = 0
    gc.collect()



    

    minibatch_size = 30


    errors_shape = ((minibatch_size)/nt,nt,4)

    target_zero = np.zeros(errors_shape);

    batches = range(num_batches) #a list of the indices all samples of 10 frames
    training_losses = []
    print "begin training"

    for e in range(num_epochs): #execute some epochs
        print "start epoch " + str(e)
        np.random.shuffle(batches) #randomize sample order
        print batches
        bi = 0
        while bi < num_batches: #for each minibatch
            batch_num = batches[bi]
            
            #batch = samples[batch_size*batch_num:batch_size*(batch_num + 1)] #construct a list of indices of samples for this batch
            print " batch: " + str(batch_num)
            fname = DATA_DIR + "train"+str(batch_num * batch_size) +"_" + str((batch_num + 1) * batch_size) + ".mat"
            print fname
        
        
            bat = np.array(loadmat(fname)['d'])/255.0 
            
            closs= 0 #tracks loss over this batch
        
            for j in range(batch_size/minibatch_size):
                
                
                

                for i in range(minibatch_size/nt):
                    
                    
    


                    X_train =  np.zeros((minibatch_size/nt,nt, 128, 160,    3))
                
                    X_train[i,:,:,:,:] = bat[j*minibatch_size + i*nt:j*minibatch_size +(i+1)*nt]
                    
                    
                    
                #X_train[0,:,:,:,:] = f[0:batch_size/2]
                #X_train[1,:,:,:,:] = f[batch_size/2:]#initialize X_test before we load values
                
                #print target_zero.shape
                ind = 0 #index in X_train
            
                X_train = np.transpose(X_train, (0, 1, 4, 2, 3)) #permute dimensions so prednet can accept it
                #print X_train.shape
                l = model.train_on_batch(X_train, target_zero)
                closs +=l
                print "minibatch " ,j,"/",batch_size/minibatch_size, " mean loss:", closs/j, "               \r",
            
 
                break_loop=False
                for layer in model.layers:
                
                    weights = layer.get_weights()
                    totnans = 0
                    totinfs = 0
                    for arr in weights:
                    
                        totnans +=  np.sum(np.isnan(np.array(arr)))
                        totinfs +=  np.sum(np.isinf(np.array(arr)))
                    if totnans > 0:
                        #print totnans
                        #print  "batch"+ str(batch_num) +"minibatch" + str(j)
                        #print "NAN weights encountered, reload last weights, shuffle and retry"
                        #reload previous batch weights
                        
                        model.load_weights(out_file)
                        
                        #shuffle
                        batches = batches[bi:]
                        np.random.shuffle(batches)
                        bi =0
                        num_batches -=bi
                        
                        print batches
                        #exit batch and retry
                        break_loop=True
                        #for now, exit to show it's happeneing
                        #sys.exit()
                        
                        """ ani_frame (bat[j*minibatch_size:(j+1)*minibatch_size],
                                   5,
                                   "batch"
                                   + str(batch_num) +
                                   "minibatch"
                                   + str(j),
                                   30)
                        sys.exit()
                        """
                        
                    #print "nans and infs"
                
                    #print totnans
                    #print totinfs
                    
                    #print "batch: " + str(batch_num) + " loss: " + str(l)

                if break_loop:
                    break
            print ""
            closs = closs/(batch_size/minibatch_size)
            training_losses.append(closs)
            
            bi+=1
            model.save_weights(out_file, overwrite=True)
            with open(out_file+".json",'w') as f:
                f.write(model.to_json())
                f.close()
            plt.plot(training_losses)
            plt.savefig("vim2_weights/lossese2-6.png")
            
            
            print "batch: " + str(batch_num) + " Loss this batch: " + str(closs)
            
    


    
np.save("vim2_weights/losseslastbatch.npy",training_losses)
json_string = model.to_json()
model.save_weights(out_file, overwrite=True)
with open(out_file + ".json", "w") as f:
    f.write(json_string)



