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
#PTH_CSV = "../../../opt/data/driving_log.csv"
#PTH_IMG = "../../../opt/data/IMG/"
PTH_CSV = "../data_bhvr_cln/driving_log.csv"
PTH_IMG = "../data_bhvr_cln/IMG/"

lines=[]

with open(PTH_CSV) as csvfile:
    has_header = csv.Sniffer().has_header(csvfile.read(1024))
    csvfile.seek(0)  # Rewind.
    reader = csv.reader(csvfile)
    if has_header:
        next(reader)  # Skip header row.
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for batch_sample in batch_samples:
                #print(line)
                src_path = batch_sample[0]
                #print(src_path)
                f_name = src_path.split('/')[-1]
                #print(f_name)
                #break
                current_path = PTH_IMG +f_name
                image = ndimage.imread(current_path)
                images.append(image)
                measurement = float(line[3])
                measurements.append(measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)


# Creating the model

from keras.models import Sequential, Model
from keras.layers import Cropping2D,Lambda
from keras.layers import Dense,Flatten

model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(160, 320,3)))

# trim image to only see section with road
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))

# Add rest of the model

model.add(Flatten(input_shape=(3,160,320)))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_lines),\
                    validation_data=validation_generator,\
                    validation_steps=len(validation_lines), epochs=3)

# Saving the mode

model.save('model.h5')

