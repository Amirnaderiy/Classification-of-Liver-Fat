# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 11:45:48 2023

@author: Asus
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import os , shutil
from tensorflow.python.client import device_lib
from keras import optimizers
from tensorflow.keras import layers, models
from keras import metrics
from keras import layers
from keras import models
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD , Adam
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import math

# Data : Make Directories

import os
import random
from sklearn.model_selection import train_test_split
import cv2

# Set the path to the directory containing the images
original_dataset_dir = 'path/to/original_dataset_dir'

# Define the four groups based on the filename
group0 = [filename for filename in os.listdir(original_dataset_dir) if filename.startswith('Group0')]
group1 = [filename for filename in os.listdir(original_dataset_dir) if filename.startswith('Group1')]
group2 = [filename for filename in os.listdir(original_dataset_dir) if filename.startswith('Group2')]
group3 = [filename for filename in os.listdir(original_dataset_dir) if filename.startswith('Group3')]

# Define a function to load the images and resize them to a specific size
def load_and_resize_image(filename, target_size):
    img = cv2.imread(os.path.join(original_dataset_dir, filename))
    img = cv2.resize(img, target_size)
    return img

# Define the target size for the images
target_size = (224, 224)

# Load the images for each group and resize them
group0_images = [load_and_resize_image(filename, target_size) for filename in group0]
group1_images = [load_and_resize_image(filename, target_size) for filename in group1]
group2_images = [load_and_resize_image(filename, target_size) for filename in group2]
group3_images = [load_and_resize_image(filename, target_size) for filename in group3]

# Define the labels for each group
group0_labels = [0] * len(group0_images)
group1_labels = [1] * len(group1_images)
group2_labels = [2] * len(group2_images)
group3_labels = [3] * len(group3_images)

from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the EfficientNetB7 model
conv_base = EfficientNetB7(weights='imagenet', 
                      include_top=False, input_shape=(224,224,3))

conv_base.summary()

# Network architecture
model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))

print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))

# Set the paths for the training, validation, and test directories
train_dir = 'path/to/train_dir'
validation_dir = 'path/to/validation_dir'
test_dir = 'path/to/test_dir'

# Define the batch size for training and validation
batch_size = 32

# Define the number of classes
num_classes = 4

# Define the data generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.5, 1.5],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)



test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    shuffle=True,
    batch_size=32,
    class_mode='categorical'
)

validation_generator=val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size= 16 ,
    class_mode='categorical'
)
    
    

def step_decay(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 50
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])


    
history = model.fit(train_generator, 
                    steps_per_epoch=len(train_generator), 
                    epochs=250, 
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator),
                    callbacks=[lr_scheduler])

    
history_dict=history.history
accuracy_values=history_dict['accuracy']
val_accuracy_values=history_dict['val_accuracy']
epochs=range(1,len(accuracy_values)+1)
plt.plot (epochs,accuracy_values,'bo',label='Training Accuracy')
plt.plot (epochs,val_accuracy_values,'r',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show
history_dict=history.history
accuracy_values=history_dict['loss']
val_accuracy_values=history_dict['val_loss']
epochs=range(1,len(accuracy_values)+1)
plt.plot (epochs,accuracy_values,'b*',label='Training Loss')
plt.plot (epochs,val_accuracy_values,'r',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show

print ('This is the number of trainable weights'
       'after freezing the conv base :', len (model.trainable_weights)) 
set_trainable=False
for layer in conv_base.layers:
    if layer.name=='block5_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False
print ('This is the number of trainable weights'
       'after freezing the conv base :', len (model.trainable_weights))     

   
        
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.0005),
              metrics=["accuracy"])
history =model.fit_generator(
    train_generator,
    steps_per_epoch= 12 ,
    epochs=100  ,
    validation_data=validation_generator,
    validation_steps=5)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='categorical')

#test_loss , test_acc = model.evaluate_generator(test_generator,steps=6)
#print ('test acc : ', test_acc)
pred2= model.predict_generator(test_generator, 83 // 16)
predicted_class_indices=np.argmax(pred2,axis=1)
labels=(validation_generator.class_indices)
labels2=dict((v,k) for k,v in labels.items())
predictions=[labels2[k] for k in predicted_class_indices]

test_loss , test_acc = model.evaluate_generator(test_generator,steps=6)
print ('test acc : ', test_acc)



