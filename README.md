## Behaviorial Cloning, Project 3
### Project Resubmission

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

#### Contents include:

* model.py: code for training the model; using python 3.6, keras 1.2.1  
* drive.py: Udacity provided script to drive the car in autonomous mode
* model.h5- a trained Keras model
* writeup.md - a report writeup file
* video_resub.mp4 - a video recording of the simulator vehicle driving autonomously 

#### Goals 
* Use the simulator to collect driving behavior data
* Build, a convolution neural network in Keras that predicts steering angles from driving images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

#### 1. Model Architecture 

Uses and adaptation of the Nvidia architecture described ![here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The dropouts and keep probability values were modeled after ![this architecture](https://github.com/bhumphrey0x20/Behavior-Cloning/edit/master/writeup.md). For parameter tuning the Adam optimizer was used to automatically adjust the learning rate. Epoches were adjusted such that the Validation Loss was near it's lowest value, at 4 epochs (see Model Fitting below). Relu activation functions were used after the first to convolutional layers. Max pooling was used after the the first three layers.


#### Table 1. Nvidia-type CNN Architeture

| Layer No  | Functions     |Dimensions                                   |
|-----------|---------------|---------------------------------------------|
|Layer1:    |Normalization  |(img/255 - 0.5)  |
|           |Conv           |kernel = 5x5|
|           |Max_Pool       |kernel = 2x2, strides = 2                  |
|Layer2:    |Conv           |kernel = 5x5|
|           |Max_Pool       |kernel = 2x2, strides = 2                  |
|Layer3:    |Conv           |kernel = 5x5|
|Layer4:    |Conv           |kernel = 3x3|
|Layer5:    |Conv           |kernel = 3x3|
|Layer 6:   |Flattening |                    |    
|           |Dropout  | keep prob = 0.5
|Layer7:    |Fully Connect  |
|           |Dropout  | keep prob = 0.5
|Layer8:    |Fully Connect  |
|           |Dropout  | keep prob = 0.5
|Layer9:    |Fully Connect  |
|           |Dropout  | keep prob = 0.5
|Layer10:    |Fully Connect  |
|           |Dropout  | keep prob = 0.5
|Layer11:    |Fully Connect  |




#### Data Collection 

![Udacity driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) was used to successfully train the model. The data was preprocessed to more equally distribute the driving angles and reduce the number of zero-valued angles, data augmentation was added, and generators were used to successfully train the model. Augmentation included a combination of flipping, image brightening, and shifting. Additionally the images were converted to YUV color space, cropped, and resized to 64x64.


##### Fig 1. Histogram of Steering Angles Before Preprocessing
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Behavior-Cloning/master/images/hist_data.png" height="240" width="320" />
##### Fig 2. Histogram of Steering Angles After Preprocessing
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Behavior-Cloning/master/images/hist_preproc_data.png" height="240" width="320" />

##### Fig 3. Image Augmentation: Original, Flipped, Brightened/Darkened, Shifted Images
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Behavior-Cloning/master/images/augmentation.png" height="344" width="1395" />


The data was normalization according to the equation ( img/255 - 0.5 ). Images were cropped by 50 rows from the top and 20 rows from the bottom. Only center camera angles were used. The data was shuffled and 20% of the data was split apart and used for validation, while the remaining 80% was used for training.


The model was trained for 4 epochs, using a batch size = 125. Loss was calculated using MSE.  

### Video Implementation 

<a href="https://youtu.be/zvdoz4i2Xrw" target="_blank"><img src="https://i9.ytimg.com/vi/zvdoz4i2Xrw/1.jpg?sqp=CJzUxNkF&rs=AOn4CLAXGuzDIqP3bDoc15pD-ocNd1B18w" alt="Behavioral Cloning Video" width="240" height="180" border="10" /></a>
