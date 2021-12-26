import os
import cv2
import numpy as np
import argparse 

from utils import datagen, iou_seg
from losses.losses import dice_coef, iou_seg

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import normalize

from sklearn.model_selection import train_test_split

input_dir = "/content/train_256"
target_dir = "/content/train_groundtruth_256"
img_w = 256
img_h = 256
n_label = 1

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        
    ]
)

images = [cv2.imread(i, 0) for i in input_img_paths]
masks = [cv2.imread(i, 0) for i in target_img_paths]

#Normalize images
images = np.expand_dims(normalize(np.array(images), axis=1),3)
#Rescale to 0 to 1.
masks = np.expand_dims((np.array(masks)),3) /255.

x_train, x_test, y_train, y_test = train_test_split(
	images, 
	masks, 
	test_size = 0.10, 
	random_state = 0
	)

#train_generetor, val_generator = datagen(x_train, x_test, y_train, y_test)

IMG_HEIGHT = x_train.shape[1]
IMG_WIDTH  = x_train.shape[2]
IMG_CHANNELS = x_train.shape[3]
num_labels = 1  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 8
#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
from focal_loss import BinaryFocalLoss
from tensorflow.keras.optimizers import Adam

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-1,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True)

from keras_unet_collection import models, losses
###############################################################################
#Model 1: Unet
help(models.unet_2d)

model_Unet = models.unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=1, 
                           activation='RELU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool='max', unpool='nearest',  
                           name='unet')

tf.keras.utils.plot_model(
    model_Unet,
    to_file="/content/drive/MyDrive/visualizations/model_Unet.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
    layer_range=None,
    show_layer_activations=False,
)

model_Unet.compile(loss='binary_crossentropy',
	optimizer=Adam(learning_rate=lr_schedule), 
	metrics=[losses.dice_coef, iou_seg], )

print(model_Unet.summary())

csv_logger = tf.keras.callbacks.CSVLogger('/content/unet.csv')
checkpoint = model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/unet_weigths.h5',
    verbose=1,
    save_best_only=True)

callbacks = [csv_logger, checkpoint]

start1 = datetime.now() 

Unet_history = model_Unet.fit(x_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(x_test, y_test ), 
                    shuffle=False,
                    epochs=50,
                    callbacks=callbacks)

stop1 = datetime.now()
#Execution time of the model 
execution_time_Unet = stop1-start1
print("UNet execution time is: ", execution_time_Unet)

model_Unet.save('mitochondria_unet_collection_UNet_50epochs.h5')

##############################################################################
#Model 2: Attention U-net

help(models.att_unet_2d)

model_att_unet = models.att_unet_2d((256, 256, 3),
	[64, 128, 256, 512],
	n_labels=1,
	stack_num_down=2, stack_num_up=2,
	activation='ReLU', atten_activation='ReLU',
	attention='add', output_activation=None, 
	batch_norm=True, pool=False, unpool='bilinear', 
	name='attunet')


model_att_unet.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = lr_schedule), 
              metrics=[losses.dice_coef, iou_seg])

print(model_att_unet.summary())

csv_logger = tf.keras.callbacks.CSVLogger('/content/att_unet_2d.csv')
checkpoint = model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/att_unet_2d_weigths.h5',
    verbose=1,
    save_best_only=True)

callbacks = [csv_logger, checkpoint]

start2 = datetime.now() 

att_unet_history = model_att_unet.fit(x_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(x_test, y_test ), 
                    shuffle=False,
                    epochs=50,
                    callbacks=callbacks)

stop2 = datetime.now()
#Execution time of the model 
execution_time_att_Unet = stop2-start2
print("Attention UNet execution time is: ", execution_time_att_Unet)

model_att_unet.save('mitochondria_unet_collection_att_UNet_50epochs.h5')

#######################################################################
#Model 3: UNet 3+
help(models.r2_unet_2d)

model_Unet_3plus = models.unet_3plus_2d((256, 256, 3), n_labels=1, filter_num_down=[64, 128, 256, 512], 
                             filter_num_skip='auto', filter_num_aggregate='auto', 
                             stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                             batch_norm=True, pool='max', unpool=False, deep_supervision=True, name='unet3plus')

model_Unet_3plus.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = lr_schedule), 
              metrics=[losses.dice_coef, losses.iou_seg])
model_Unet_3plus.summary()

csv_logger = tf.keras.callbacks.CSVLogger('/content/unet_3plus_2d.csv')
checkpoint = model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/unet_3plus_2d_weigths.h5',
    verbose=1,
    save_best_only=True)

callbacks = [csv_logger, checkpoint]

start3 = datetime.now() 

model_Unet_3plus_history = model_Unet_3plus.fit(x_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(x_test, y_test ), 
                    shuffle=False,
                    epochs=50,
                    callbacks=callbacks)

stop3 = datetime.now()
#Execution time of the model 
execution_time_model_Unet_3plus = stop3-start3
print("UNet+ execution time is: ", execution_time_model_Unet_3plus)

model_r2_Unet.save('mitochondria_unet_collection_model_Unet_3plus_50epochs.h5')
####################################################################################
#Model 4: Recurrent Residual (R2) U-Net
help(models.r2_unet_2d)

model_r2_Unet = models.r2_unet_2d((256, 256, 3),[64, 128, 256, 512], n_labels=1,
                          stack_num_down=2, stack_num_up=1, recur_num=2,
                          activation='ReLU', output_activation='Softmax', 
                          batch_norm=True, pool='max', unpool='nearest', name='r2unet')


model_r2_Unet.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = lr_schedule), 
              metrics=[losses.dice_coef, losses.iou_seg])

model_r2_Unet.summary()

csv_logger = tf.keras.callbacks.CSVLogger('/content/r2_unet_2d.csv')
checkpoint = model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/r2_unet_2d_weigths.h5',
    verbose=1,
    save_best_only=True)

callbacks = [csv_logger, checkpoint]

start4 = datetime.now() 

r2_Unet_history = model_r2_Unet.fit(x_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(x_test, y_test ), 
                    shuffle=False,
                    epochs=50,
                    callbacks=callbacks)

stop4 = datetime.now()
#Execution time of the model 
execution_time_r2_Unet = stop4-start4
print("R2 UNet execution time is: ", execution_time_r2_Unet)

model_r2_Unet_from_scratch.save('mitochondria_unet_collection_r2_UNet_50epochs.h5')

####################################################################################
#Model 5: U2-Net

model_u2net = models.u2net_2d((256, 256, 3), n_labels=2, 
                        filter_num_down=[64, 128, 256, 512], filter_num_up=[64, 64, 128, 256], 
                        filter_mid_num_down=[32, 32, 64, 128], filter_mid_num_up=[16, 32, 64, 128], 
                        filter_4f_num=[512, 512], filter_4f_mid_num=[256, 256], 
                        activation='ReLU', output_activation=None, 
                        batch_norm=True, pool=True, unpool=True, deep_supervision=True, name='u2net')

model_u2net.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate = lr_schedule), 
              metrics=[losses.dice_coef, losses.iou_seg])

model_u2net.summary()

csv_logger = tf.keras.callbacks.CSVLogger('/content/u2net_2d.csv')
checkpoint = model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/u2net_2d_weigths.h5',
    verbose=1,
    save_best_only=True)

callbacks = [csv_logger, checkpoint]

start5 = datetime.now() 

model_u2net_history = model_u2net.fit(x_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(x_test, y_test ), 
                    shuffle=False,
                    epochs=50,
                    callbacks=callbacks)

stop5 = datetime.now()
#Execution time of the model 
execution_time_u2net = stop5-start5
print("U2-Net execution time is: ", execution_time_u2net)

model_u2net.save('mitochondria_unet_collection_u2net_50epochs.h5')
