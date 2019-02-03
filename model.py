# Model.py
# Sample data location /opt/carnd_p3/data/
# Model.py
# Sample data location /opt/carnd_p3/data/

import csv
from scipy import ndimage
import numpy as np

# Importing training data 
PTH_CSV = "../../../opt/data/driving_log.csv"
PTH_IMG = "../../../opt/data/IMG/"
#PTH_CSV = "../data_bhvr_cln/driving_log.csv"
#PTH_IMG = "../data_bhvr_cln/IMG/"

lines=[]

with open(PTH_CSV) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines[1:]:
	#print(line)
	src_path = line[0]
	#print(src_path)
	f_name = src_path.split('/')[-1]
	#print(f_name)
	#break
	current_path = PTH_IMG +f_name
	image = ndimage.imread(current_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


# Creating the model

from keras.models import Sequential
from keras.layers import Dense,Flatten

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True, nb_epoch=7)

# Saving the mode

model.save('model.h5')

