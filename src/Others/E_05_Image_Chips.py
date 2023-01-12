# import copy
import os
import numpy as np
from pyrsgis import raster
from pyrsgis.ml import imageChipsFromFile

# define the file names
feature_file = r"C:\Users\pablo\OneDrive - AGFOREST\05_TRABAJO\01_AMIANTO\01_IMPROVEMENTS\image_semantic_segmentation\workspace_2\images\8-15-2020_Ortho_MS.tif"
label_file = r"C:\Users\pablo\OneDrive - AGFOREST\05_TRABAJO\01_AMIANTO\01_IMPROVEMENTS\image_semantic_segmentation\workspace_2\masks\Total_Segment.tif"

# create feature chips using pyrsgis
features = imageChipsFromFile(feature_file, x_size=7, y_size=7)

""" Update: 29 May 2021
Since I added this code chunk later, I wanted to make least 
possible changes in the remaining sections. The below line changes
the index of the channels. This will be undone at a later stage.
"""
features = np.rollaxis(features, 3, 1)
print("Hola")
# read the label file and reshape it
ds, labels = raster.read(label_file)
labels = labels.flatten()

# check for irrelevant values (we are interested in 1s and non-1s)
labels = (labels == 1).astype(int) # added on 29 Aug 2021

# print basic details
print('Input features shape:', features.shape)
print('\nInput labels shape:', labels.shape)
print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))

# Go up one directory
os.chdir('..\\..')

# Save the arrays as .npy files
np.save('CNN_7by7_features.npy', features)
np.save('CNN_7by7_labels.npy', labels)