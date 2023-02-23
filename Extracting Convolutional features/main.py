# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 00:39:39 2023

@author: Asus
"""
from keras.applications import InceptionResNetV2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_selection import VarianceThreshold
import os, shutil
from keras.preprocessing.image import ImageDataGenerator



conv_base=InceptionResNetV2(weights='imagenet',
        include_top=False,
        input_shape=(636,444,3))
#conv_base.summary ()
# Extract features


datagen = ImageDataGenerator(rescale=1./255)
batch_size = 8

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 18, 12, 1536))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count,4))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
        target_size=(636,444,),
        batch_size = batch_size,
        shuffle=False,
        class_mode='categorical')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels
    
train_features, train_labels = extract_features(train_dir, 464)  # Agree with our small dataset size
test_features, test_labels = extract_features(test_dir, 86)

X_train, y_train = train_features.reshape(464,18*12*1536), train_labels
X_test, y_test = test_features.reshape(86,18*12*1536), test_labels

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))    
new_X_train=sel.fit_transform(X_train)
new_X_test=sel.fit_transform(X_test)

rf_features = np.concatenate((X_train, X_test))
rf_labels = np.concatenate((y_train, y_test))
new_X_rf=sel.fit_transform(rf_features)
