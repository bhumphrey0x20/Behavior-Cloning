# **Behavioral Cloning**
## **Project 3 Udacity Self-Driving Car Nanodegree** 

## Writeup 
#### ( Project Resubmission )
---

### Goals 
* Use the simulator to collect driving behavior data
* Build, a convolution neural network in Keras that predicts steering angles from driving images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

### Changes For Project Resubmittion

Changes since original submission include:
1) Preprocessing of data to remove the number of zero-valued steering angles
2) Data augmentation 
3) Use of generators 
4) Update of CNN Architecture including use of dropouts 

### Files Submitted & Code Quality

#### 1. Included Files for Submission

This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video_resub.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The python script model.py is updated code used for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Changes made since the original submission include 

### Model Architecture and Training Strategy

#### 1. Model Architecture 

The original architecture used was adapted from a former Udacity student's Tensorflow [Traffic Sign Classifiers project](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)- described as an adaptation of a Sermanet/LeCunn classifier. However this was changed to something very similar to the Nvidia architecture described ![here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). The dropouts and keep probability values were modeled after ![this architecture](https://github.com/bhumphrey0x20/Behavior-Cloning/edit/master/writeup.md)  


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


For parameter tuning the Adam optimizer was used to automatically adjust the learning rate. Epoches were adjusted such that the Validation Loss was near it's lowest value, at 4 epochs (see Model Fitting below). Relu activation functions were used after the first to convolutional layers. Max pooling was used after the the first three layers. 

#### 2. Data Collection, Preprocessing, and Training Strategy

The original data used for training and validation included two passes around the track (counter-clockwise) of straight line driving and one pass of straight line driving around the track in the opposite directon (clockwise). Additional data was collected recovering from various tack features (e.g. the bridge, "dirt pull-off"), however this did not help to produce a successful model. 

Ultimately, ![Udacity driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) was used to successfully train the model. The data was preprocessed to more equally distribute the driving angles, data augmentation was added, and generators were used to successfully train the model.

#### 3. Data Preprocessing

A histogram of the steering angles shows a large number of zero values. Figure 1 shows the steeing angle distribution from the Udacity data set. The number of zero-valued steering angles was 4373, biasing it toward straight line driving. Ideas from ![here](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff) and ![here](https://medium.com/@fromtheast/you-dont-need-lots-of-data-udacity-behavioral-cloning-6d2d87316c52) were employed to reduce the number of zero-valued angles. 

Image path and and steering angles were read from the csv file (lines 239-262) and sorted/classified using function preprocess_data() (lines 78-125). The preprocessing steps shuffled, and appended angle values and image pathes to lists based on the value of the angle. Angles between -015 and 0.15 were classified as center angles; angles < -0.15 were classified as left angles; and angles > 0.15 were classified as right angles. Next, the center angles were split using `train_test_split()`. 2% were were retained and stored in a `center_angle` list. 98% were stored in a `center_mix_angle` list and redistributed to the left and right lists. Image paths were also appended to their corresponding center, left, and right lists.

For redistribution, angles < 0, from the `center_mix_angles` list, were classified as left angles, a random number between 0 and 0.25 was subtracted from the angle value, and appended to the angle list. The path list was "flagged" and appended to the left image path list. The same was done for angles > 0. A random number was added to the right angle value and appended to the right list. The image path was flagged accordingly. Angles = 0 were discarded. Finally, the original `center_angle` list was appened to the updated left and right lists. This preprocessing step reduced the number of zero-valued steering angles to 445 (see Figure 2).

Flags appended to the image path list were used to indicate which camera angle to use from the Udacity image set (center, left, or right), during training of the model. For paths appended to the left angle list (angle < 0), a "2" was appended to indicate right camera. Conversely, a "1" was appended to right image paths, to indicate the use of left cameras. If no flag was appended then the default center imgae was used (see lines 106-116 and 141-144). This was also a form of data augmentation.


##### Fig 1. Histogram of Steering Angles Before Preprocessing
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Behavior-Cloning/master/images/hist_data.png" height="240" width="320" />
##### Fig 2. Histogram of Steering Angles After Preprocessing
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Behavior-Cloning/master/images/hist_preproc_data.png" height="240" width="320" />


Preprocessed data was shuffled and 20% of the data was split apart and used for validation testing, while the remaining 80% was used for training (line 302).

#### 4. Data Augmentation

Ideas for data augmentation are described ![here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9). These include fliping the images, brightening/darkening the images, shifting or translating the images left or right, or using a combination of flipping and brightening. These processes were performed inside the generator function `generate_batch()` (lines 162-226). Flipping used `numpy.fliplr()` and changed the sign of the angle. `brighten_image()` (lines 62-73) converted image to HSV and multiplied the v-channel by a random number. (Note: color conversion to HSV used the YUV image as if it were BGR, then converts the image back to BGR. This was done by accident, but worked!) `image_shift()` translated the image randomly left or right and multiplied the angle by a random `shift_factor`.

Additional, augmentation was performed in the function `generate_data()` (lines 133-158) and included converting the images to YUV color space, cropping the images by 50 rows from the top and 20 rows from the bottom; and resizing the image to 64x64.

##### Fig 3. Image Augmentation: Original, Flipped, Brightened/Darkened, Shifted Images
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Behavior-Cloning/master/images/augmentation.png" height="344" width="1395" />



#### 5. Generator Functions

To add training and testing data to the model and reduce memory usage two generator function were used. The first, `generate_data()` (line 133-158), randomly chose an angle and image from the angle and camera lists. Camera image (left, right, or center) was determined by the camera image flag (discussed above). The image was converted to YUV color space, cropped, and resized to 64x64. 

The second generator function `generate_batch()` (line 162-266) was used to create batches used for model training and testing. The function used images generated from `generate_data()` and randomly selected a data augmentation step( flipping, brightening, flipping and brightening, or shifting).

A `generate_data()` and `generate_batch()` function were created for both training data and testing data. 


#### Model Fitting

For training and testing 4 epochs were used with a batch size of 125, and a samples per epoch of 20,000. These were determined by experimentally. Loss was determined using Mean Squared Error. 

#### drive.py File

For autonomous driving, the drive.py file was modified to convert the imput image from to YUV color space, crop the image, and resize the image to 64x64 (lines 65-68).


### Video Implementation 

<a href="https://youtu.be/zvdoz4i2Xrw" target="_blank"><img src="https://i9.ytimg.com/vi/zvdoz4i2Xrw/1.jpg?sqp=CJzUxNkF&rs=AOn4CLAXGuzDIqP3bDoc15pD-ocNd1B18w" alt="Behavioral Cloning Video" width="240" height="180" border="10" /></a>
