#Procedimiento

# ¿Lo tinenes ya entrenado el modelo?
# True/False
# If True:
# Put the model to Use
# Put the image to predict

# If False:
# Put the image
# Put the mask
# Select type of model you want to Use
# Select the variables to start the training: epochs and batch size
# The new model created will be the one to UserWarning

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import rasterio
from rasterio.plot import show
from PIL import Image

workspace = os.path.abspath('workspace_2/')
image_to_show = os.path.abspath(f"{workspace}/images/part_14.tif")
print(image_to_show)
patch_size = 256
image_to_show = os.path.abspath(f"{workspace}/masks/part_14.tif")

###############################################
##                  cv2
###############################################
# IMREAD_UNCHANGED = -1,
# IMREAD_GRAYSCALE = 0,
# IMREAD_COLOR = 1,
# IMREAD_ANYDEPTH = 2,
# IMREAD_ANYCOLOR = 4,
# IMREAD_LOAD_GDAL = 8,
# IMREAD_REDUCED_GRAYSCALE_2 = 16,
# IMREAD_REDUCED_COLOR_2 = 17,
# IMREAD_REDUCED_GRAYSCALE_4 = 32,
# IMREAD_REDUCED_COLOR_4 = 33,
# IMREAD_REDUCED_GRAYSCALE_8 = 64,
# IMREAD_REDUCED_COLOR_8 = 65,
# IMREAD_IGNORE_ORIENTATION = 128

temp_img = cv2.imread(image_to_show, 128) #3 channels / spectral bands
print(f'dtype: {temp_img.dtype}, shape: {temp_img.shape}, min: {np.min(temp_img)}, max: {np.max(temp_img)}')
temp_img = cv2.imread(image_to_show,1)
temp_img_2 = cv2.imread(image_to_show,-1)
# temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
# print(temp_img.shape)
# print(temp_img.max())

print(f'dtype: {temp_img.dtype}, shape: {temp_img.shape}, min: {np.min(temp_img)}, max: {np.max(temp_img)}')
print(f'dtype: {temp_img_2.dtype}, shape: {temp_img_2.shape}, min: {np.min(temp_img_2)}, max: {np.max(temp_img_2)}')

# SIZE_X = (temp_img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
# SIZE_Y = (temp_img.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
# print(SIZE_X)
# print(SIZE_Y)
# print(temp_img)
# image = Image.fromarray(temp_img)
# image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
# #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
# image = np.array(image)
# # print(image)

# """
# temp_img = cv2.imread(image_to_show,1)
# print(temp_img.shape)
# print(temp_img.max())

# import numpy as np
# temp_img = cv2.imread(image_to_show, -1)
# print(f'dtype: {temp_img.dtype}, shape: {temp_img.shape}, min: {np.min(temp_img)}, max: {np.max(temp_img)}')
# plt.imshow(temp_img)
# plt.show()

# image = cv2.imread(image_to_show, 1)  #Read each image as BGR
# print(f'dtype: {image.dtype}, shape: {image.shape}, min: {np.min(image)}, max: {np.max(image)}')

# image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# print(f'dtype: {image.dtype}, shape: {image.shape}, min: {np.min(image)}, max: {np.max(image)}')


# image = cv2.imread(image_to_show, cv2.IMREAD_ANYDEPTH)
# img_cv_16bit  = image.view(np.int16)
# print(f'dtype: {img_cv_16bit.dtype}, shape: {img_cv_16bit.shape}, min: {np.min(img_cv_16bit)}, max: {np.max(img_cv_16bit)}')
# img_cv_8bit = np.clip(img_cv_16bit // 16, 0, 255).astype(np.uint8)
# print(f'dtype: {img_cv_8bit.dtype}, shape: {img_cv_8bit.shape}, min: {np.min(img_cv_8bit)}, max: {np.max(img_cv_8bit)}')



# # Mat test1(1000, 1000, CV_16U, Scalar(400));
# # imwrite("test.tiff", test1);
# # Mat test2 = imread("stam.tiff", CV_LOAD_IMAGE_ANYDEPTH);
# # cout << test1.depth() << " " << test2.depth() << endl;
# # cout << test2.at<unsigned short>(0,0) << endl;

# # gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
# # print(gray.max())

# # norm_image = cv2.normalize(temp_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# # print(temp_img.max())

# # plt.imshow("", temp_img, cmap="Greys")
# # plt.show()

# # result = cv2.normalize(temp_img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_16S)
# # result = temp_img/255
# # info = np.iinfo(temp_img.dtype) # Get the information of the incoming image type
# # data = temp_img.astype(np.float64) / info.max # normalize the data to 0 - 1
# # data = 255 * data # Now scale by 255
# # img = data.astype(np.uint8)
# # cv2.imshow("Window", img)
# # cv2.waitKey(0)
# # print(result.shape)
# # print(result.max())
# # plt.imshow(result)
# # plt.show()


# # cv2.imshow("image2", temp_img)
# # cv2.waitKey(0)

# # plt.imshow(temp_img[:, :, 0]) #View each channel...
# # plt.show()

# """

# ###############################################
# ##                  rasterio
# ###############################################
# image = rasterio.open(image_to_show)
# # show((raster, 3))
# # show(raster.read(), transform=raster.transform)
# image = image.read()
# print(image.shape)
# print(f'dtype: {image.dtype}, shape: {image.shape}, min: {np.min(image)}, max: {np.max(image)}')
# shape_0 = image.shape[1]
# shape_1 = image.shape[2]
# shape_2 = image.shape[0]
# shape = (shape_0, shape_1, shape_2)

# # plt.imshow(image[1, :, :]) #View each channel...
# # plt.show()

# SIZE_X = (shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
# SIZE_Y = (shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
# print(SIZE_X)
# print(SIZE_Y)
# print(image)
# img = cv2.imread(image, 1)
# print(img)
# # image = image.astype(np.uint8)
# # print(image)
# # image = image/256
# # print(image)
# image = Image.fromarray(image)
# # image = Image.fromarray((image * 255).astype(np.uint8))

# print(image)
# image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
# #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation

# # image = np.array(image)

# # import uuid
# # import gdal
# # import os

# # fn = "/path/to/some/file.tif"
# # outfn = "/path/to/some/outfile.tif"

# # xmin, ymin, xmax, ymax = # to be decided by the user
# # xres, yres = # to be decided by the user
# # extent = [xmin, ymin, xmax, ymax]
# # if not os.path.exists(outfn):

# #     # set the warp options
# #     warpoptions = dict(
# #         outputBounds=extent, # set the extent of the output
# #         xRes=xres, yRes=yres # set the x and y resolution of the output
# #         dstSRS="EPSG:32737", # destination SRS is EPSG:32737
# #         format="VRT" # warp to VRT
# #     )

# #     # set the translate options
# #     translateoptions = dict(
# #         scaleParams=[[0,2**16,0,2**8]], # rescale data from 16 to 8 bit
# #         outputType=gdal.GDT_Byte,
# #         creationOptions=["COMPRESS=LZW","TILED=YES"] # compress and tile the output
# #     )

# #     # step 1, warp input data to 32737 in memory
# #     desc = "/vsimem/" + uuid.uuid4().hex
# #     ds = gdal.Warp(desc, fn, **warpoptions)

# #     # step 2, translate the in memory dataset and apply the scaling
# #     ds2 = gdal.Translate(outfn, ds, **translateoptions)
# #     ds2 = None

# #     # close the in memory dataset
# #     ds = None
# #     gdal.Unlink(desc)