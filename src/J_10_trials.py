import warnings
warnings.filterwarnings("ignore")

# Required libraries to work with semantic segmentation
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import rasterio as rs
from rasterio.plot import show

from osgeo import gdal
from osgeo import osr
from osgeo import ogr

from matplotlib import gridspec
import time
from patchify import patchify,unpatchify
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
from sklearn.model_selection import train_test_split
import random
import cv2
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
from C_03_smooth_tiled_predictions import predict_img_with_smooth_windowing

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

workspace = os.path.abspath('workspace/data_for_keras_aug/')
img_file_loc = os.path.abspath(f"{workspace}/train_images/train/IGN_image1.tifpatch_346.tif") 
mask_file_loc = os.path.abspath(f"{workspace}/train_masks/train/Total_Segment_1.tifpatch_346.tif")

img = cv2.imread(img_file_loc)
original_mask = cv2.imread(mask_file_loc)

input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
input_img = preprocess_input(input_img)

from keras.models import load_model
epochs = 50
batch_size = 5
model_name = f"landcover_{epochs}_epochs_RESNET_backbone_batch{batch_size}.hdf5"

model = load_model(model_name, compile=False)
                  
# size of patches
patch_size = 256
# Number of classes 
n_classes = 2

###################################################################################
#Predict using smooth blending

# Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number. 2 = 50% overlap
    nb_classes=n_classes,
    pred_func=(
        lambda img_batch_subdiv: model.predict((img_batch_subdiv))
    )
)

final_prediction = np.argmax(predictions_smooth, axis=2)
# print(final_prediction)

DirtoSaveResults = os.path.abspath(f"workspace_2/test_images/")
try:
    os.makedirs(DirtoSaveResults)
    print("folder '{}' created ", DirtoSaveResults)
except FileExistsError:
    var = 2
    # print("not")

# plt.imsave('workspace_2/test_images/try_segmented.tif', final_prediction)
im = Image.fromarray(final_prediction.astype('float32')) # (final_prediction * 255).astype(np.uint8))
im.save('workspace_2/test_images/try_segmented.tif')

img = cv2.imread('workspace_2/test_images/try_segmented.tif')
print(img)
img = img.astype(np.uint8)

plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(img )
plt.subplot(222)
# plt.title('Testing Label')
# plt.imshow(original_mask)
plt.subplot(223)
plt.title('Prediction with smooth blending')
plt.imshow(final_prediction)
plt.show()

# addBands(srcImg, names, overwrite)
# displacement(referenceImage, maxOffset, projection, patchWidth, stiffness)
# displace(displacement, mode, maxOffset)
# register(referenceImage, maxOffset, patchWidth, stiffness)
# reproject(crs, crsTransform, scale)
# sample(region, scale, projection, factor, numPixels, seed, dropNulls, tileScale, geometries)
# setDefaultProjection(crs, crsTransform, scale)


############################################################################
##
##                            REVIEW INFO IMAGE
##
############################################################################

#looking at the data for first time..
# file_loc =  os.getcwd() + "\XYZ.tif"
# mask_img = rs.open(mask_file_loc)
# geo_img = rs.open(img_file_loc)
# # show(geo_img)
# print("Count: ",geo_img.count)
# print("Height and widht: ", geo_img.height ,geo_img.width)
# print("CRS", geo_img.crs)
# print("bounds: ",geo_img.bounds)

# #visualizing all the spectral bands individually
# display_name = ["Blue", "green", "red"]
# fig = plt.figure()
# # to change size of subplot's
# fig.set_figheight(20)
# fig.set_figwidth(20)
# spec = gridspec.GridSpec(ncols=3, nrows=1,wspace=0.5,
#                          hspace=0.5)
# n_cols = 3
# for index in range(n_cols):
#     ax = fig.add_subplot(spec[index])
#     ax.title.set_text(display_name[index])
#     ax.imshow(geo_img.read(index+1))
# plt.show()

# #visualizing the distribution of bands in image
# plt.figure(figsize=(10,10))
# rs.plot.show_hist(geo_img)
# plt.show()

############################################################################
##
##                            MAKE PATCHES
##
############################################################################

############################################################################
##                            REAL IMAGE
############################################################################
### Features for the data
# with rs.open(img_file_loc, 'r') as file:
#     arr_st = file.read()
# # Data
# # arr_st = np.stack(features)
# features = np.moveaxis(arr_st, 0, -1)
# print(features.shape) #(2554, 2562, 3)

# # making an empty list
# image_dataset = []
# #I am going to divide my image into 256*256
# #make sure your image shape is divisible by the patch size or reshape your image accordingly
# patch_size = 256
# patches_img = patchify(features, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        
# for i in range(patches_img.shape[0]):
#     for j in range(patches_img.shape[1]):
#         single_patch_img = patches_img[i,j,:,:]
#         #Use minmaxscaler instead of just dividing by 255. 
#         single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
#         #single_patch_img = (single_patch_img.astype('float32')) / 255. 
#         single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
#         image_dataset.append(single_patch_img)

############################################################################
##                            MASK IMAGE
############################################################################
### Features for the data
# with rs.open(mask_file_loc, 'r') as file:
#     arr_st = file.read()
# # Data
# # arr_st = np.stack(features)
# features = np.moveaxis(arr_st, 0, -1)
# print(features.shape) #(2504, 3334, 1)

# # making an empty list
# mask_dataset = []
# #I am going to divide my image into 256*256
# #make sure your image shape is divisible by the patch size or reshape your image accordingly
# patch_size = 256
# patches_mask = patchify(features, (patch_size, patch_size, 1), step=patch_size)  #Step=256 for 256 patches means no overlap
        
# for i in range(patches_mask.shape[0]):
#     for j in range(patches_mask.shape[1]):
#         single_patch_mask = patches_mask[i,j,:,:]
#         #Use minmaxscaler instead of just dividing by 255. 
#         single_patch_mask = scaler.fit_transform(single_patch_mask.reshape(-1, single_patch_mask.shape[-1])).reshape(single_patch_mask.shape)
#         #single_patch_img = (single_patch_img.astype('float32')) / 255. 
#         single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
#         mask_dataset.append(single_patch_mask)

# print(len(image_dataset))
# print(len(mask_dataset))
# #visualizing the image to see if they are patch with correct label (no neccessary but to be sure)
# image_number = random.randint(0, len(image_dataset))
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
# plt.subplot(122)
# plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 1)))
# plt.show()

# #hexa -> decimal
# rice = '#ffd300'.lstrip('#')
# rice= np.array(tuple(int(rice[i:i+2], 16) for i in (0, 2, 4)))
# # wheat = '#267000'.lstrip('#')
# # wheat = np.array(tuple(int(wheat[i:i+2], 16) for i in (0, 2, 4)))

# #use this function to all mask_dataset values 
# def rgb_to_2D_label(label):
#     label_seg = np.zeros(label.shape,dtype=np.uint8)
#     # label_seg [np.all(label == wheat,axis=-1)] = 1
#     label_seg [np.all(label==rice,axis=-1)] = 1

#     label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels

#     return label_seg

# labels =[rice]
# n_classes = len(np.unique(labels))
# print(n_classes)
# labels_cat = to_categorical(labels, num_classes=n_classes)
# #spliting training test using sklearn library
# X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)