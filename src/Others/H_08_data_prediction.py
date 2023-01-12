from pyrsgis.ml import imageChipsFromArray
import tensorflow as tf
from pyrsgis import raster
import numpy as np
import h5py
import os

dirname = os.path.abspath(os.curdir)
# os.chdir("..")

# model_name = os.path.join(dirname, 'workspace_2/trained_models/200410_CNN_Builtup_PScore0.649_RScore0.032_FScore0.060.h5')
# model_name = model_name.replace('\\', '/')

model_name = os.path.join(dirname, 'CNN_Builtup.h5')
model_name = model_name.replace('\\', '/')
print('\n', '\n', model_name, '\n', '\n')

# Load the saved model
model = tf.keras.models.load_model(model_name)

# Load a new multispectral image
image_pedrajas = r"C:\Users\pablo\OneDrive - AGFOREST\05_TRABAJO\01_AMIANTO\02_MUNICIPIOS\Pedrajas_San_Esteban\Dataset\HR_Image\tif1.tif"
ds, featuresHyderabad = raster.read(image_pedrajas)

# Generate image chips from array
""" Update: 29 May 2021
Note that this time we are generating image chips from array
because we needed the datasource object (ds) to export the TIF file.
And since we are reading the TIF file anyway, why not use the array directly.
"""
new_features = imageChipsFromArray(featuresHyderabad, x_size=7, y_size=7)

print('Shape of the new features', new_features.shape)

# Predict new data and export the probability raster
newPredicted = model.predict(new_features)
#print("Shape of the predicted labels: ", newPredicted.shape)
newPredicted = newPredicted[:,1]

prediction = np.reshape(newPredicted, (ds.RasterYSize, ds.RasterXSize))

outFile = '200408_Pedrajas_BuiltupCNN_predicted_7by7.tif'
raster.export(prediction, ds, filename=outFile, dtype='float')