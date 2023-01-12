import os
import numpy as np
from sklearn.utils import resample
import tensorflow as tf
import random
import keras

os.chdir(r"C:\Users\pablo\OneDrive - AGFOREST\05_TRABAJO\01_AMIANTO\01_IMPROVEMENTS\image_semantic_segmentation\workspace_2")

# Load arrays from .npy files
features = np.load('CNN_7by7_features.npy')
labels = np.load('CNN_7by7_labels.npy')

# print(len(features))
# print(len(labels))
if len(labels)>len(features):
    number = len(labels) - len(features)
    labels = labels[number:]
elif len(labels)<len(features):
    number = len(features) - len(labels)
    features = features[number:]
else: 
    print("features and Labels have same lenght")

# Separate and balance the classes
built_features = features[labels==1]
built_labels = labels[labels==1]

unbuilt_features = features[labels==0]
unbuilt_labels = labels[labels==0]

print('Number of records in each class:')
print('Built: %d, Unbuilt: %d' % (built_labels.shape[0], unbuilt_labels.shape[0]))


# Downsample the majority class
unbuilt_features = resample(unbuilt_features,
                            replace = False, # sample without replacement
                            n_samples = built_features.shape[0], # match minority n
                            random_state = 2)

unbuilt_labels = resample(unbuilt_labels,
                          replace = False, # sample without replacement
                          n_samples = built_features.shape[0], # match minority n
                          random_state = 2)

print('Number of records in balanced classes:')
print('Built: %d, Unbuilt: %d' % (built_labels.shape[0], unbuilt_labels.shape[0]))

# Combine the balanced features
features = np.concatenate((built_features, unbuilt_features), axis=0)
labels = np.concatenate((built_labels, unbuilt_labels), axis=0)

# Normalise the features
eight_bits = 255.0
sixteen_bits = 65535.0

features = features / sixteen_bits
print('New values in input features, min: %d & max: %d' % (features.min(), features.max()))

# Define the function to split features and labels
def train_test_split(features, labels, trainProp=0.6):
    dataSize = features.shape[0]
    sliceIndex = int(dataSize*trainProp)
    randIndex = np.arange(dataSize)
    random.shuffle(randIndex)
    train_x = features[[randIndex[:sliceIndex]], :, :, :][0]
    test_x = features[[randIndex[sliceIndex:]], :, :, :][0]
    train_y = labels[randIndex[:sliceIndex]]
    test_y = labels[randIndex[sliceIndex:]]
    return(train_x, train_y, test_x, test_y)
  
# Call the function to split the data
train_x, train_y, test_x, test_y = train_test_split(features, labels)

""" Update: 29 May 2021
Transpose the features to channel last format.
If you have commented out the rollaxis line in the 
first place. You can comment the following two lines too.
"""
train_x = tf.transpose(train_x, [0, 2, 3, 1])
test_x = tf.transpose(test_x, [0, 2, 3, 1])

print('Reshaped split features:', train_x.shape, test_x.shape)
print('Split labels:', train_y.shape, test_y.shape)

######################################################################################################
# Create and Train the model
kSize = 7
nBands = 4

# from tensorflow.keras.layers import Conv2D
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

# Create a model
model = keras.Sequential()
model.add(Conv2D(32, kernel_size=1, padding='valid', activation='relu', input_shape=(kSize, kSize, nBands)))
model.add(Dropout(0.25))
model.add(Conv2D(48, kernel_size=1, padding='valid', activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Run the model
model.compile(loss='sparse_categorical_crossentropy', optimizer= 'rmsprop',metrics=['accuracy'])
model.fit(train_x, train_y)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Predict for test data 
yTestPredicted = model.predict(test_x)
yTestPredicted = yTestPredicted[:,1]

# Calculate and display the error metrics
yTestPredicted = (yTestPredicted>0.5).astype(int)
cMatrix = confusion_matrix(test_y, yTestPredicted)
pScore = precision_score(test_y, yTestPredicted)
# print("P-Score:", pScore)
rScore = recall_score(test_y, yTestPredicted)
# print("R-Score:", rScore)
fScore = f1_score(test_y, yTestPredicted)
# print("F-Score:", fScore)

print("Confusion matrix:\n", cMatrix)

print("\nP-Score: %.3f, R-Score: %.3f, F-Score: %.3f" % (pScore, rScore, fScore))

# Save the model inside a folder to use later
if not os.path.exists(os.path.join(os.getcwd(), 'trained_models')):
    os.mkdir(os.path.join(os.getcwd(), 'trained_models'))
    
model.save('trained_models/200410_CNN_Builtup_PScore%.3f_RScore%.3f_FScore%.3f.h5' % (pScore, rScore, fScore)) 
