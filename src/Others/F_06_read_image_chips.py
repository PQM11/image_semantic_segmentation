import os, glob
import numpy as np
from pyrsgis import raster

# Change the working directory
imageDirectory = r"C:\Users\pablo\OneDrive - AGFOREST\05_TRABAJO\01_AMIANTO\01_IMPROVEMENTS\image_semantic_segmentation\workspace_2\ImageChips"
os.chdir(imageDirectory)

# Get the number of files in the directory
nFiles = len(glob.glob('*.tif'))

# Get basic information about the image chips
ds, tempArr = raster.read(os.listdir(imageDirectory)[0])
nBands, rows, cols = ds.RasterCount, ds.RasterXSize, ds.RasterYSize

# Create empty arrays to store data later
features = np.empty((nFiles, nBands, rows, cols))
labels = np.empty((nFiles, ))

# Loop through the files, read and stack
for n, file in enumerate(glob.glob('*.tif')):
    ds, tempArr = raster.read(file)
    # Get filename without extension, split by underscore and get the label
    tempLabel = os.path.splitext(file)[0].split('_')[-1]

    features[n, :, :, :] = tempArr
    labels[n] = tempLabel

# check for irrelevant values (we are interested in 1s and non-1s)
labels = (labels == 1).astype(int) # added on 29 Aug 2021
    
print('Input features shape:', features.shape)
print('\nInput labels shape:', labels.shape)
print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))