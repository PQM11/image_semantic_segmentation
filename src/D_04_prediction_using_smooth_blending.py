# https://youtu.be/0W6MKZqSke8
"""
Prediction using smooth tiling as described here...
https://github.com/Vooban/Smoothly-Blend-Image-Patches
"""

# import sys

# if not sys.warnoptions:
#     import warnings
#     warnings.simplefilter("ignore")
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from C_03_smooth_tiled_predictions import predict_img_with_smooth_windowing

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

img_folder = os.path.abspath("workspace_2/images/IGN_image1.tif")#IGN_image1.tifpatch_346.tif
mask_folder = os.path.abspath("workspace_2/masks/Total_Segment_1.tif")#Total_Segment_1.tifpatch_346.tif
# img_folder = os.path.abspath("workspace_2/images/IGN_image1.tifpatch_346.tif")
# mask_folder = os.path.abspath("workspace_2/masks/Total_Segment_1.tifpatch_346.tif")
# print('\n', '\n', img_folder, '\n', '\n')

img = cv2.imread(img_folder)

input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
input_img = preprocess_input(input_img)

original_mask = cv2.imread(mask_folder)

# original_mask = original_mask[:,:,0]  #Use only single channel...
#original_mask = to_categorical(original_mask, num_classes=n_classes)

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

# data = np.random.randint(0, 255, (10,10)).astype(np.uint8)
# print(type(data))
# print(type(final_prediction))
# im = Image.fromarray(data)
# data.save('workspace_2/test_images/test.tif')

# Save prediction and original mask for comparison
# plt.imsave('workspace_2/test_images/MDC_try_segmented.tif', final_prediction)
im = Image.fromarray((final_prediction * 255).astype(np.uint8))
im.save('workspace_2/test_images/MDC_try_segmented.tif')
# plt.imsave('workspace_2/test_images/MDC_try_mask.jpg', original_mask)

# import rasterio

# with rasterio.open('workspace_2/test_images/MDC_try_segmented.tif', 'w', 
#                     driver='GTiff',
#                     height= final_prediction.shape[0],
#                     width= final_prediction.shape[1],
#                     dtype=final_prediction.dtype,
#                     count=1) as dst:
#     dst.write(final_prediction,1)

# with rasterio.open('workspace_2/test_images/MDC_try_mask.tif', 'w') as dst:
#     dst.write(original_mask)

print("\n", "\n", final_prediction.dtype,"\n", "\n")
# img = cv2.imread(final_prediction)
# cv2.imwrite('workspace_2/test_images/test.tif', img)

###################

plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(img)
# plt.subplot(222)
# plt.title('Testing Label')
# plt.imshow(original_mask)
plt.subplot(223)
plt.title('Prediction with smooth blending')
plt.imshow(final_prediction)
plt.show()

#############################