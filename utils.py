import cv2
import tensorflow as tf
from tf.keras.preprocessing.image import ImageDataGenerator
from skimage import io

def read_data(path, save_path):
    img = io.imread(path)
    result = [io.imsave(save_path+"{}.tif".format(i), img[i]) for i in range(len(img))]

#dispatch images

'''
filepath = ''
xdims = 255
ydims = 255
'''

def get_path(filepath)
    input_img_paths = sorted(
        [
            os.path.join(filepath, fname)
            for fname in os.listdir(filepath)
            
        ]

    )
    return input_img_paths

def get_patches(input_img_paths, xdims, ydims):
    images = cv2.imread(input_img_paths)
    i=165
    for img in images:
        j=0
        i+=1
        for r in range(0,img.shape[0],xdims):
            for c in range(0,img.shape[1],ydims):
                j+=1
                cv2.imwrite("{}_{}".format(i, j) + f"_img{r}_{c}.png",img[r:r+xdims, c:c+ydims,:])

#datagenerator
def datagen(x_train, y_train, x_test, y_test):
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         fill_mode='reflect')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(x_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)

    images_gen=image_datagen.flow(x_train,batch_size=4,shuffle=True, seed=seed)
    mask_gen=mask_datagen.flow(y_train,batch_size=4,shuffle=True, seed=seed)



    # Creating the validation Image and Mask generator

    image_datagen.fit(x_test, augment=True, seed=seed)
    mask_datagen.fit(y_test, augment=True, seed=seed)

    images_gen_val=image_datagen.flow(x_test,batch_size=4,shuffle=True, seed=seed)
    mask_gen_val=mask_datagen.flow(y_test,batch_size=4,shuffle=True, seed=seed)

    train_generator = zip(images_gen, mask_gen)
    val_generator = zip(images_gen_val, mask_gen_val)
    return train_generator, val_generator

    import tensorflow.keras.backend as K

def iou_seg(y_true, y_pred, dtype=tf.float32):
    """
    Inersection over Union (IoU) loss for segmentation maps. 
    
    iou_seg(y_true, y_pred, dtype=tf.float32)
    
    ----------
    Rahman, M.A. and Wang, Y., 2016, December. Optimizing intersection-over-union in deep neural networks for 
    image segmentation. In International symposium on visual computing (pp. 234-244). Springer, Cham.
    
    ----------
    Input
        y_true: segmentation targets, c.f. `keras.losses.categorical_crossentropy`
        y_pred: segmentation predictions.
        
        dtype: the data type of input tensors.
               Default is tf.float32.
        
    """

    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    area_intersect = tf.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))
    
    area_true = tf.reduce_sum(y_true_pos)
    area_pred = tf.reduce_sum(y_pred_pos)
    area_union = area_true + area_pred - area_intersect
    
    return tf.math.divide_no_nan(area_intersect, area_union)