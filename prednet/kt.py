from keras import backend as K
with K.tf.device("/gpu:1"):
    import kitti_train
