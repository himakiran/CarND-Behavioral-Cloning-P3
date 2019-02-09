# **Behavioral Cloning** 

## Self Driving Car Nano Degree

### CHUNDI HIMAKIRAN KUMAR

---

**Behavioral Cloning Project**

This report consists of the steps taken to reach the solution in the form of model.py
which is used to generate model.h5 that allows the car to successfully complete Track one.

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture 

#### 1. An appropriate model architecture has been employed as discussed below

We have used the NVIDIA model architecture for this project. The model consists of 

* 1st layer of CNN: 24 fliters, 5x5 kernel, 2x2 stride
* 2nd layer of CNN: 36 filters, 5x5 kernel, 2x2 stride
* 3rd layer of CNN: 48 filters, 5x5 kernel, 2x2 stride
* 4th layer of CNN: 64 filters, 3x3 kernel, non-stride
* 5th layer of CNN: 64 filters, 3x3 kernel, non-stride
* Dropout(dropout 50% of connection)
* Flatten
* 1st fully connected layer: 100 neurons
* 2nd fully connected layer: 50 neurons
* 3rd fully connected layer: 10 neurons
* Output:(Measurement)


#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer in order to reduce overfitting as shown above. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

#### 4. Appropriate training data

The Sample training data was used to derive the final solution though in the experimental trials, the trained data on track
one was used. Finally however the solution could be achieved using the sample data set.



### Solution Design Approach

* 1. Firstly we tried to derive the solution using the sample data set and the NVIDIA model architecture without any change.
When that did not succeed, we decided to first visualize the training data.
* 2. The visualization of the sample training data set which was obtained by a 80% split of the sample data provided can be observed in the Jupyter Notebook included.
* 3. We see that there is an excessive values of the zero steering angle measurement. Hence we decided to balance the data. By trial and error we settled on 80/100 ie all steering angles whose qty was less than 80 were raised up to 80 and all those whose sum was beyond 100 brought down to 100. Post balanicng we 
* 4. When even after this did not bring about much success, we decided to work with our own trained data. However we realized that the trained data was inferior to the sample data provided as the training was done using the keyboard and the results were not encouraging.
* 5. Hence we decided to augment the data after balancing it so that we could get more data for training. Augmentation was 
carried out using
    * (a) All images were flipped and the negative of the original measurement was recorded for the flipped image.
    * (b) Left camera images were included and the mesaurement value of the original image was decremented by a fixed qty which
    was yet another parameter for fine tunign the alogorithm.
    * (c) Right camera images were included and the mesaurement value of the original image was incremented by a fixed qty which
    was yet another parameter for fine tuning the alogorithm.
* 6. Finally the data was run through the model and success was acheieved with 0.4 as the parameter above and training for two epochs.
    

#### 2. Final Model Architecture

* The final model architecture consisted of the following
    * (a) Reading in the data and splitting it to get 80% training and 20% validation data.
    * (b) Balancing the data with 80/100 which was a paramater value we got after extensive trial and error.
    * (c) Augmenting the data as explained above
    * (d) Feeding the data to unaltered NVIDIA CNN and compiled using the ADAM optimizer with default values.
    * (e) Training the model for two epochs.

#### 3. A Note on Generators and Google Cloud Compute

* 1. Generators in python help us deal with large memory operations. However the side effect they had in this project was that
the Udacity GPU was extremely slow when processing the code with the generator function. A lot of Udacity GPU hours were wasted
in training as the GPU would take very long time and the workspace would go idle and we would lose all the training.

* 2. The above led us to use Google Cloud Compute which gives 300 USD worth of Cloud compute resources. We initailly tried to
train the model using Google TPU but failed as the code had to be rewritten for the TPU. Later we trained the model extensively
using a Ubuntu VM with a P100 Tesla GPU . Thanks to this URL 
 [Google_Cloud_VM_How-To](https://mc.ai/easy-set-up-of-gpu-enabled-tensorflow-on-google-cloud-or-any-other-virtual-machine/)
 
#### 4. Jupyter notebook
* 1. I found it much useful to run parts of code on the Jupyter notebook to both examine and visualize data and also
do the trial and error with fine tuning paarameters. I am including a copy of the jupyter notebook saved in pdf format.