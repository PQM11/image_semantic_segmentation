
"""
Code can be modified to work with any other dataset.
Tasks achieved.
1. Read large images and corresponding masks, divide them into smaller patches.
And write the patches as images to the local drive.  
2. Save only images and masks where masks have some decent amount of labels other than 0. 
Using blank images with label=0 is a waste of time and may bias the model towards 
unlabeled pixels.
 * Maybe not neccesary to do it in my case
3. Divide the sorted dataset from above into train and validation datasets. 
4. Move some folders and rename appropriately if you want to use 
ImageDataGenerator from keras. 
"""
import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
import glob

from matplotlib import pyplot as plt
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# import tifffile as tiff
from PIL import Image
# import tensorflow as tf
# from tensorflow import keras
# import segmentation_models as sm
# from tensorflow.keras.metrics import MeanIoU
import random
import rasterio
from rasterio.plot import show

print("Iniciamos")
workspace = os.path.abspath('workspace/')
print("\n","\n","\n","\n", workspace, "\n","\n","\n","\n")
#Quick understanding of the dataset
# raster = rasterio.open(os.path.abspath(f"{workspace}/images/IGN_image.tif"))
# # show((raster, 3))
# show(raster.read(), transform=raster.transform)

temp_mask = cv2.imread(os.path.abspath(f"{workspace}/masks/Total_Segment_1.tif")) #3 channels but all same. 
labels, count = np.unique(temp_mask[:,:,0], return_counts=True) #Check for each channel. All chanels are identical
print("\n","\n","\n","\n", "Labels are: ", labels, " and the counts are: ", count, "\n","\n","\n","\n")

#Now, crop each large image into patches of 256x256. Save them into a directory 
#so we can use data augmentation and read directly from the drive. 
root_directory = workspace

patch_size = 256

#Read images from respective 'images' subdirectory
#As all images are of different size we have 2 options, either resize or crop
#But, some images are too large and some small. Resizing will change the size of real objects.
#Therefore, we will crop them to a nearest size divisible by 256 and then 
#divide all images into patches of 256x256x3.

img_dir = os.path.abspath(root_directory+"/images/")
print(img_dir)
for path, subdirs, files in os.walk(img_dir):
    print(path)
    dirname = path.split(os.path.sep)[-1]
    #print(dirname)
    images = os.listdir(path)  #List of all image names in this subdirectory
    print(images)
    for i, image_name in enumerate(images):  
        if image_name.endswith(".tif"):

            # print("\n","\n","\n","\n", path+"/"+image_name, "\n","\n","\n")          
            with rasterio.open(path+"/"+image_name, 'r') as file:
                arr_st = file.read()
            # Data
            # arr_st = np.stack(features)
            features = np.moveaxis(arr_st, 0, -1)
            print(features.shape)

            #Extract patches from each image
            print("Now patchifying image:", path+"/"+image_name)
            patches_img = patchify(features, (256, 256, 3), step=256)  #Step=256 for 256 patches means no overlap
            
            ##########################
            print(patches_img.shape)
            ##########################
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i,j,:,:]
                    #Use minmaxscaler instead of just dividing by 255. 
                    # single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                    single_patch_img = (single_patch_img.astype('float32')) / 255. 
                    single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                    # print(type(single_patch_img))
                    # image_dataset.append(single_patch_img)
                    DirtoSaveResults = os.path.abspath(f'{workspace}/256_patches/images/')
                    try:
                        os.makedirs(DirtoSaveResults)
                        print("folder '{}' created ", DirtoSaveResults)
                    except FileExistsError:
                        var = 2
                        # print("not")
                    root_directory = os.path.abspath("/256_patches/images/")
                    # cv2.imshow("", y.astype('float32'))
                    cv2.imwrite(DirtoSaveResults +"/" + image_name+"patch_"+str(i)+str(j)+".tif", single_patch_img)#.astype(np.uint8)
  
#Now do the same as above for masks
#For this specific dataset we could have added masks to the above code as masks have extension png
workspace = os.path.abspath('workspace/')
mask_dir = os.path.abspath(workspace+"/masks/")
print(mask_dir)
for path, subdirs, files in os.walk(mask_dir):
    print(path)
    dirname = path.split(os.path.sep)[-1]
    #print(dirname)
    mask = os.listdir(path)  #List of all image names in this subdirectory
    print(mask)
    for i, mask_name in enumerate(mask):  
        if mask_name.endswith(".tif"):

            # print("\n","\n","\n","\n", path+"/"+image_name, "\n","\n","\n")          
            with rasterio.open(path+"/"+mask_name, 'r') as file:
                arr_st = file.read()
            # Data
            # arr_st = np.stack(features)
            features = np.moveaxis(arr_st, 0, -1)
            print(features.shape)
            #Extract patches from each image
            print("Now patchifying image:", path+"/"+mask_name)
            patches_mask = patchify(features, (256, 256,1), step=256)  #Step=256 for 256 patches means no overlap
            
            ##########################
            print(patches_mask.shape)
            ##########################
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i,j,:,:]
                    #Use minmaxscaler instead of just dividing by 255. 
                    # single_patch_mask = scaler.fit_transform(single_patch_mask.reshape(-1, single_patch_mask.shape[-1])).reshape(single_patch_mask.shape)
                    single_patch_mask = (single_patch_mask.astype('float32')) # / 255. 
                    single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
                    # print(type(single_patch_img))
                    # image_dataset.append(single_patch_img)
                    # plt.imshow(single_patch_mask) #View each channel...
                    # plt.show()
                    DirtoSaveResults = os.path.abspath(f'{workspace}/256_patches/masks/')
                    try:
                        os.makedirs(DirtoSaveResults)
                        print("folder '{}' created ", DirtoSaveResults)
                    except FileExistsError:
                        var = 2
                        # print("not")
                    root_directory = os.path.abspath("/256_patches/masks/")
                    # cv2.imshow("", y.astype('float32'))
                    cv2.imwrite(DirtoSaveResults +"/" + mask_name +"patch_"+str(i)+str(j)+".tif", single_patch_mask)#.astype(np.uint8)

            # mask_name = mask_name.split('.')
            # mask_name = mask_name[0]        
            # mask = cv2.imread(path+"/"+mask_name, 0)  #Read each image as Grey (or color but remember to map each color to an integer)
            # SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
            # SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
            # mask = Image.fromarray(mask)
            # mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
            # #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
            # mask = np.array(mask)             
   
            # #Extract patches from each image
            # print("Now patchifying mask:", path+"/"+mask_name)
            # patches_mask = patchify(mask, (256, 256), step=256)  #Step=256 for 256 patches means no overlap

            # ##########################
            # print("HOLAAAAAAAAAAAAAAAA!!!!!!!!!")
            # print(patches_mask.shape)
            # ##########################

            # for i in range(patches_mask.shape[0]):
            #     for j in range(patches_mask.shape[1]):
                    
            #         single_patch_mask = patches_mask[i,j,:,:]
            #         #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
            #         #single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.
            #         plt.imshow(single_patch_mask) #View each channel...
            #         plt.show()
            #         DirtoSaveResults = os.path.abspath(f'{workspace}/256_patches/masks/')
            #         try:
            #             os.makedirs(DirtoSaveResults)
            #             print("folder '{}' created ", DirtoSaveResults)
            #         except FileExistsError:
            #             var = 2
            #             # print("folder {} already exists".format(root_directory+"256_patches/masks/"))

            #         cv2.imwrite(DirtoSaveResults +"/" +
            #                    mask_name+"patch_"+str(i)+str(j)+".tif", single_patch_mask)


train_img_dir = os.path.abspath(f"{workspace}/256_patches/images/")
train_mask_dir = os.path.abspath(f"{workspace}/256_patches/masks/")

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))

img_num = random.randint(0, num_images-1)

rand_list=[]
n=10
for i in range(n):
    rand_list.append(random.randint(0, num_images-1))
print(rand_list)

for i in rand_list:
    print(i)
    img_for_plot = cv2.imread(train_img_dir+'/'+img_list[i], 1)
    # img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

    # img_for_plot = rasterio.open(train_img_dir+'/'+img_list[img_num])
    # img_for_plot = img_for_plot.read()

    # mask_for_plot = rasterio.open(train_img_dir+'/'+img_list[img_num])
    # mask_for_plot = mask_for_plot.read()

    mask_for_plot =cv2.imread(train_mask_dir+'/'+msk_list[i], -1)

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(img_for_plot, cmap='gray')
    plt.title('Image')
    # plt.show()
    plt.subplot(122)
    plt.imshow(mask_for_plot, cmap='gray')
    plt.title('Mask')
    plt.show()


###########################################################################

#Now, let us copy images and masks with real information to a new folder.
# real information = if mask has decent amount of labels other than 0. 

useless=0  #Useless image counter
for img in range(len(img_list)):   #Using t1_list as all lists are of same size
    img_name=img_list[img]
    mask_name = msk_list[img]
    print("Now preparing image and masks number: ", img)
      
    temp_image=cv2.imread(train_img_dir+'/'+img_list[img], 1)
   
    temp_mask=cv2.imread(train_mask_dir+'/'+msk_list[img], -1)
    #temp_mask=temp_mask.astype(np.uint8)
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0]/counts.sum())) > 0.05:  #At least 5% useful area with labels that are not 0
        print("Save Me")
        DirtoSaveResults_1 = os.path.abspath(f'{workspace}/256_patches/images_with_useful_info/images/')
        DirtoSaveResults_2 = os.path.abspath(f'{workspace}/256_patches/images_with_useful_info/masks/')
        try:
            os.makedirs(DirtoSaveResults_1)
            os.makedirs(DirtoSaveResults_2)
        except FileExistsError:
            var = 2

        cv2.imwrite(os.path.abspath(f"{workspace}/256_patches/images_with_useful_info/images/"+img_name), temp_image)
        cv2.imwrite(os.path.abspath(f"{workspace}/256_patches/images_with_useful_info/masks/"+mask_name), temp_mask)
        
    else:
        print("I am useless")   
        useless +=1

print("Total useful images are: ", len(img_list)-useless)  #175
print("Total useless images are: ", useless) #1715
###############################################################
#Now split the data into training, validation and testing. 

"""
Code for splitting folder into train, test, and val.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 
pip install split-folders
"""

import splitfolders  # or import split_folders
import os

workspace = os.path.abspath('workspace/')
input_folder = os.path.abspath(f"{workspace}/256_patches/images_with_useful_info/")
output_folder = os.path.abspath(f"{workspace}/data_for_training_and_testing/")
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None) # default values
########################################

root_directory = os.path.abspath('workspace/')
train_data_image_folder = output_folder + "/" + "train" + "/" + "images"
train_data_mask_folder = output_folder + "/" + "train" + "/" + "masks"

val_data_image_folder = output_folder + "/" + "val" + "/" + "images"
val_data_mask_folder = output_folder + "/" + "val" + "/" + "masks"

print("Let´s create folder distribution for keras augmentation")
try:
    first = os.path.abspath("workspace/data_for_keras_aug/train_images/train/")
    second = os.path.abspath("workspace/data_for_keras_aug/train_masks/train/")
    third = os.path.abspath("workspace/data_for_keras_aug/val_images/val/")
    fourth = os.path.abspath("workspace/data_for_keras_aug/val_masks/val/")
    os.makedirs(first)
    os.makedirs(second)
    os.makedirs(third)
    os.makedirs(fourth)
except FileExistsError:
    var = 2

import shutil
import os

# path to destination directory
dest_dir = os.path.abspath(f"{root_directory}/data_for_keras_aug/train_images/train")
# getting all the files in the source directory
files = os.listdir(train_data_image_folder)
# try:
#     shutil.rmtree(dest_dir)
# except FileExistsError:
#     var = 2
shutil.copytree(train_data_image_folder, dest_dir, dirs_exist_ok=True)

# path to destination directory
dest_dir = os.path.abspath(f"{root_directory}/data_for_keras_aug/train_masks/train/")
# getting all the files in the source directory
files = os.listdir(train_data_mask_folder)
shutil.rmtree(dest_dir)
shutil.copytree(train_data_mask_folder, dest_dir)

# path to destination directory
dest_dir = os.path.abspath(f"{root_directory}/data_for_keras_aug/val_images/val/")
# getting all the files in the source directory
files = os.listdir(val_data_image_folder)
shutil.rmtree(dest_dir)
shutil.copytree(val_data_image_folder, dest_dir)

# path to destination directory
dest_dir = os.path.abspath(f"{root_directory}/data_for_keras_aug/val_masks/val/")
# getting all the files in the source directory
files = os.listdir(val_data_mask_folder)
shutil.rmtree(dest_dir)
shutil.copytree(val_data_mask_folder, dest_dir)


#Now manually move folders around to bring them to the following structure.
"""
Your current directory structure:
Data/
    train/
        images/
            img1, img2, ...
        masks/
            msk1, msk2, ....
    val/
        images/
            img1, img2, ...
        masks/
            msk1, msk2, ....
        
Copy the folders around to the following structure... 
Data/
    train_images/
                train/
                    img1, img2, img3, ......
    
    train_masks/
                train/
                    msk1, msk, msk3, ......
                    
    val_images/
                val/
                    img1, img2, img3, ......                
    val_masks/
                val/
                    msk1, msk, msk3, ......
      
                    
"""
