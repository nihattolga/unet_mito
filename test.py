import pandas as pd
from losses.losses import dice_coef, iou_seg
from utils import iou_seg

'''
# convert the history.history dict to a pandas DataFrame and save as csv for
# future plotting or use saved ones
unet_history_df = pd.DataFrame(Unet_history.history) 
unet_plus_history_df = pd.DataFrame(Unet_plus_history.history) 
att_unet_history_df = pd.DataFrame(att_unet_history.history) 

unet_from_scratch_history_df = pd.DataFrame(Unet_from_scratch_history.history) 
r2_Unet_from_scratch_history_df = pd.DataFrame(r2_Unet_from_scratch_history.history) 
att_unet_from_scratch_history_df = pd.DataFrame(att_unet_from_scratch_history.history) 

with open('unet_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)
    
with open('unet_plus_history_df.csv', mode='w') as f:
    unet_plus_history_df.to_csv(f)

with open('att_unet_history_df.csv', mode='w') as f:
    att_unet_history_df.to_csv(f)    

with open('unet_from_scratch_history_df.csv', mode='w') as f:
    unet_from_scratch_history_df.to_csv(f)    
    
with open('r2_Unet_from_scratch_history_df.csv', mode='w') as f:
    r2_Unet_from_scratch_history_df.to_csv(f)    

with open('att_unet_from_scratch_history_df.csv', mode='w') as f:
    att_unet_from_scratch_history_df.to_csv(f)        
'''

#######################################################################
#Check history plots, one model at a time
history = Unet_history
'''
history = Unet_plus_history
history = att_unet_history
history = Unet_from_scratch_history
history = r2_Unet_from_scratch_history
history = att_unet_from_scratch_history
'''
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['dice_coef']
#acc = history.history['accuracy']
val_acc = history.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Dice')
plt.plot(epochs, val_acc, 'r', label='Validation Dice')
plt.title('Training and validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()

#######################################################


'''
model = model_Unet
model = model_Unet_plus
model = model_att_unet

model = model_Unet_from_scratch
model = model_r2_Unet_from_scratch
model = model_att_unet_from_scratch
'''

from keras_unet_collection.activations import GELU
model = model_Unet
#Load one model at a time for testing.
model = tf.keras.models.load_model('/content/mitochondria_unet_collection_UNet_50epochs.h5', compile=False, custom_objects={'GELU': GELU})


import random
test_img_number = random.randint(0, x_test.shape[0]-1)

test_img = x_test[test_img_number]
ground_truth=y_test[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()


#IoU for a single image
from tensorflow.keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(ground_truth[:,:,0], prediction)
print("Mean IoU =", IOU_keras.result().numpy())

'''
#Calculate IoU and average
IoU_values = []
for img in range(0, x_test.shape[0]):
    temp_img = x_test[img]
    ground_truth=y_test[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    #print(IoU)
    


df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU) 
'''

#IoU and Dice average
IoU_values = []
dice_values = []
for img in range(0, x_test.shape[0]):
    temp_img = x_test[img]
    ground_truth=y_test[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5)
    
    IoU = iou_seg(ground_truth[:,:,0], prediction)
    IoU_values.append(IoU.numpy())

    dice = losses.dice_coef(ground_truth[:,:,0], prediction)
    dice_values.append(dice.numpy())

df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU) 

df = pd.DataFrame(dice_values, columns=["dice"])
df = df[df.dice != 1.0]    
mean_dice = df.mean().values
print("Mean dice is: ", mean_dice) 