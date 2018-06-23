# **Behavioral Cloning**
## **Project 3 Udacity Self-Driving Car Nanodegree** 

## Writeup ( Resubmission )
---

### Goals 
* Use the simulator to collect driving behavior data
* Build, a convolution neural network in Keras that predicts steering angles from driving images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

#### 1. Included Files for Submission

This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The python script model.py is updated code used for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Changes made since the original submission include 

### Model Architecture and Training Strategy

#### 1. Model Architecture 

The architecture used was adapted from a former Udacity student's Tensorflow [Traffic Sign Classifiers project](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)- described as an adaptation of a Sermanet/LeCunn classifier. The model includes normalization via the `Lamdba()` function using the equation (img/255 - 0.5), three convolutional layers and a singer linear layer (see lines 347-371 in model.py). Some adjustments were made for the resubmission. The current architeture includes, a Normalizing layer, a 3x3 convolutional layer followed by a maxpooling. Next, a 1x1 convolutional is followed by a 5x5 convolution and a maxpooling. Then, another 1x1 convolution is followed by another 5x5 convolution. The output of layer 2 and layer 3 are flattened and concatenated and passed through a dropout with a keep probablity of 0.8. Finally, a single fully connected layer is performed. 



| Layer No  | Functions     |Dimensions                                   |
|-----------|---------------|---------------------------------------------|
|Layer1:    |Normalization  |(img/255 - 0.5)  |
|           |Conv           |kernel = 3x3, strides = 2 |
|           |Max_Pool       |kernel = 3x3, strides = 2                  |
|Layer2:    |Conv           |kernel = 1x1, strides = 1 |
|           |Conv           |kernel = 5x5, strides = 1 |
|           |Max_Pool       |kernel = 2x2, strides = 3 |
|Layer3:  |Conv           |kernel = 5x5, strides=2   |
|Flatten:    |Merge of Layer 3 and Layer 2 MaxPool |                    |    
|           |Dropout  | keep prob = 0.8
|Layer4:    |Fully Connect  |

An adaptation of the Nvidia architecture discussed in class was tested however, the training and validation losses were much higher that the architecture described above, therefore it was not used. 

For parameter tuning the Adam optimizer was used to automatically adjust the learning rate. Epoches were adjusted such that the Validation Loss was near it's lowest value, at 5 epochs (see Model Fitting below). 

Softmax activation functions were tested instead of Relu's, usng various epochs (5,10,15) and testing with and without dropouts, this generally resulting in underfitting and poor autonomous driving, therefore they were completely removed. 

#### 2. Data Collection, Preprocessing, and Training Strategy

The original data used for training and validation included two passes around the track (counter-clockwise) of straight line driving and one pass of straight line driving around the track in the opposite directon (clockwise). Additional data was collected recovering from various tack features (e.g. the bridge, "dirt pull-off"), however this did not help to produce a successfull model. 

Ultimately, ![Udacity driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) was used to successfully train the model. The data was preprocessed to more equally distribute the driving angles, data augmentation was added,  and generators were used to successfully train the model.

#### 3. Data Preprocessing

A histogram of the steering angles shows a large number of zero values. Figure 1 shows the steeing angle distribution from the Udacity data set. The number of zero-valued steering angles was 4373, biasing it toward straight line driving. Ideas from ![here](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff) and ![here](https://medium.com/@fromtheast/you-dont-need-lots-of-data-udacity-behavioral-cloning-6d2d87316c52) were employed to reduce the number of zero-valued angles. 

Image path and and steering angles were read from the csv file (lines 239-262) and sorted/classified using function preprocess_data() (lines 78-125). The preprocessing steps shuffled, and appended angle values and image pathes to lists based on the value of the angle. Angles between -015 and 0.15 were classified as center angles and images; angles < -0.15 were classified as left angles and images; and angles > 0.15 were classified as right angles and images. Next, the center angles were split using `train_test_split()` and 98% of the center angles were redistributed to the left and right lists. 

For redistribution, angles < 0 were classified as left angles and a random number between 0 and 0.25 was subtracted to the angle value and appended to the angle list. The path list was "flagged" and appended to the left image path list. The same was done for angles > 0. A random number was added to the right angle value and appended to the right list. The image path was flagged accordingly. Angles = 0 were discarded. Then the original center list was appened to the new left and right lists. This preprocessing step reduced the number of zero-valued steering angles to 445 (see Figure 2).

Flags appended to the image path list were used to indicate which camera angle to use, from the Udacity image set, during training of the model : center, left, or right. For paths appended to the left angle list (angle < 0), a "2" was appended to indicate right camera. Conversely, a "2" was appended to right image paths, to indicate the use of left cameras. This was also a form of data augmentation.


##### Fig 1. Histogram of Steering Angles Before Preprocessing
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Behavior-Cloning/master/images/hist_data.png" height="240" width="320" />
##### Fig 2. Histogram of Steering Angles After Preprocessing
<img src="https://raw.githubusercontent.com/bhumphrey0x20/Behavior-Cloning/master/images/hist_preproc_data.png" height="240" width="320" />


Preprocessed data was shuffled and 20% of the data was split apart and used for validation testomg, while the remaining 80% was used for training (line 302).

#### 4. Data Augmentation

Ideas for data augmentation are described ![here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9). These include fliping image, brightening/darkening (here on out called brightening) image, shifting or translating the image left or right or a combination of flipping and brightening. These processes were performed inside the generator function `generate_batch()` (lines 162-226). Flipping used `numpy.fliplr()` and changed the sign of the angle. `brighten_image()` (lines 62-73) converted image to HSV and multiplied the v-channel by a random number and converting image back to BGR color space. `image_shift()` translated the image randomly left or right and multiplied the angle by a random `shift_factor`.

Additional, augmentation was performed in the function `generate_data()` (lines 133-158) and included converting image to YUV space cropped image by 50 rows from the top and 20 rows from the bottom; and resizing the image to 64x64.

#### 5. Generator Functions

To add training and testing data to the model and reduce memory usage two generator function were used. The first, `generate_data()` (line 133-158), randomly chooses an angle and image from the angle and camera lists. Camera image (left, right, or center) was determined by the camera image flag (discussed above). The image was converted to YUV color space, cropped, and resized to 64x64. 

The second generator function `generate_batch()` (line 162-266) was used to create batches use for model training. The function take images generated from `generate_data()` and randomly choses a data augmentation step( flipping, brightening, flipping and brightening, or shifting).

A `generate_data()` and `generate_batch()` function were created for both training data and testing data. 



#### Model Fitting

For training and testing eight epochs were used with a batch size of 125, and a samples per epoch of 20,000. These were determined by experimentally. Loss was determined using Mean Squared Error. 

#### drive.py File

For autonomous driving, the drive.py file was modified to convert the imput image from to YUV color space, crop the image, and resize the image to 64x64 (lines 65-68).
