# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 23:39:09 2017

@author: Chris
"""

import os
import csv
import csv
import cv2
import numpy as np
import tensorflow as tf
import sklearn
#tf.python.control_flow_ops = tf


samples = []
with open('D:\\2017-10-14_udacity_driving_log_v3\\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=128):
    num_samples = len(samples)
   # print ('samlpes', samples)
    while 1: # Loop forever so the generator never terminates
        #sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.1
            for batch_sample in batch_samples:
                    for i in range(3):
                        current_path = batch_sample[i]
                        image = cv2.imread(current_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)
                        
                        if i is 0:
                            angle = float(batch_sample[3])
                            angles.append(angle)                             
                            image=cv2.flip(image,1) #flip
                            images.append(image)
                            angles.append(angle)
                        
                        if i is 1:
                            angle = float(batch_sample[3]) + correction
                            angles.append(angle)
                            image=cv2.flip(image,1) ##flip
                            images.append(image)
                            angles.append(-angle)
                            
                        elif i is 2:
                            angle = float(batch_sample[3]) - correction
                            angles.append(angle)
                            image=cv2.flip(image,1) ##flip
                            images.append(image)
                            angles.append(-angle)
                            

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
#            print ("All data successfully loaded into memory")
#            print ("X_data shape", X_train.shape)
#            print ("y_data shape", y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)
            
            

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)


# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160, 320, 3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255 - 0.5))
#model.add(Convolution2D(24, 8, 8, subsample=(4,4),activation='relu'))

model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples*3*2), validation_data=validation_generator,
            nb_val_samples=len(validation_samples*3), nb_epoch=7)

model.save('test_normalized_3images_nvmodel_corr0.1_ep7.h5')
print ('model successfully saved!')

from keras.models import Model
import matplotlib.pyplot as plt


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()