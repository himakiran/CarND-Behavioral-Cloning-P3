# Model.py
# Sample data location /opt/carnd_p3/data/
# Model.py
# Sample data location /opt/carnd_p3/data/
# https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zipunzip 

import os
import csv
from scipy import ndimage
import numpy as np
import sklearn

# Importing training data 
PTH_CSV = "../../../opt/carnd_p3/chk_trg_data/driving_log.csv"
PTH_IMG = "../../../opt/carnd_p3/chk_trg_data/IMG/"
#PTH_CSV = "../data_bhvr_cln/driving_log.csv"
#PTH_IMG = "../data_bhvr_cln/IMG/"

lines=[]

with open(PTH_CSV) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#print("Total length of data : {}".format(len(lines)))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

# http://jeffwen.com/2017/07/14/behavioral_cloning

# Distibuting data to correct the imabalance
# We shall reduce the data points with zero as the steering angle and increase the other data points 
# which are in minority.
# As a baseline we require minimum 80 and maximum 100 for the training data set. 
# so all data points below 80 will be brought up to 80 and those above 100 will be reduced to 100.

def  balance_data(data,min_reqd,max_reqd):
    data_output = data.copy()
    
    ## create histogram to know what needs to be added
    steering_angles = np.asarray(data_output[:,3], dtype='float')
    #print(len(np.unique(steering_angles)))
    num_hist, index_hist = np.histogram(steering_angles, np.unique(steering_angles))
    #print(len(num_hist))
    #print(len(index_hist))
    to_be_added = np.empty([1,7])
    to_be_deleted = np.empty([1,1])
    
    for i in range(1, len(num_hist)):
        if num_hist[i-1]<min_reqd:

            ## find the index where values fall within the range 
            match_index = np.where((steering_angles>=index_hist[i-1]) & (steering_angles<index_hist[i]))[0]

            ## randomly choose up to the minimum needed
            need_to_add = data_output[np.random.choice(match_index,min_reqd-num_hist[i-1]),:]
            
            to_be_added = np.vstack((to_be_added, need_to_add))

        elif num_hist[i-1]>max_reqd:
            
            ## find the index where values fall within the range 
            match_index = np.where((steering_angles>=index_hist[i-1]) & (steering_angles<index_hist[i]))[0]
            
            ## randomly choose up to the minimum needed
            to_be_deleted = np.append(to_be_deleted, np.random.choice(match_index,num_hist[i-1]-max_reqd))

        

    ## delete the randomly selected observations that are overrepresented and append the underrepresented ones
    data_output = np.delete(data_output, to_be_deleted, 0)
    data_output = np.vstack((data_output, to_be_added[1:,:]))
    
    return data_output

# Balance Training data
train_lines_balanced = balance_data(np.array(train_lines),13,20)
# Checking 
#print("Total length of balanced Training data : {}".format(len(train_lines_balanced)))
#print("Total length of balanced Validation data : {}".format(len(validation_lines_balanced)))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for batch_sample in batch_samples:
                #print(batch_sample)
                src_path = batch_sample[0]
                #print(src_path)
                f_name = src_path.split('/')[-1]
                #print(f_name)
                #break
                current_path = PTH_IMG +f_name
                image = ndimage.imread(current_path)
                # Appending original image
                # images.append(image)
                # measurement = float(line[3])
                # measurements.append(measurement)
                #appending flipped image
                #images.append(np.fliplr(image))
                #measurements.append(-measurement)
                # appending left camera image and steering angle with offset
                # src_path = batch_sample[1]
                # f_name = src_path.split('/')[-1]
                # current_path = PTH_IMG +f_name
                # image = ndimage.imread(current_path)
                # images.append(image)
                # measurements.append(measurement+0.4)
                # # appending right camera image and steering angle with offset
                # src_path = batch_sample[2]
                # f_name = src_path.split('/')[-1]
                # current_path = PTH_IMG +f_name
                # image = ndimage.imread(current_path)
                # images.append(image)
                # measurements.append(measurement-0.3)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_lines_balanced, batch_size=40)
validation_generator = generator(validation_lines, batch_size=40)


# Creating the model

from keras.models import Sequential, Model
from keras.layers import Cropping2D,Lambda
from keras.layers import Dense,Flatten,Dropout,MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint




# Adding a Nvidia  model to train on the sample data for completing track 1

model = Sequential()
# trim image to only see section with road
#model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

# Adding checkpoint
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_lines_balanced)*1,\
                    validation_data=validation_generator,\
                    validation_steps=len(validation_lines)*1, epochs=1,callbacks=[checkpoint])

# Saving the mode

model.save('modelchk.h5')