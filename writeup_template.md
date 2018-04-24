# **Behavioral Cloning**
## **Project 3 Udacity Self-Driving Car Nanodegree** 

## Writeup
---

The goals 
* Use the simulator to collect driving behavior data
* Build, a convolution neural network in Keras that predicts steering angles from driving images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
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

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Architecture 

The architecture used was adapted from a former Udacity student's Tensorflow [Traffic Sign Classifiers project](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)- itself an adaptation of Sermanet/LeCunn classifier. It uses three convolutional layer and a singer linear layer. After the first two convolutional layers, the outputs are passed through a Relu and a MaxPooling node. The output of the third convolution is flattened and concatenated with the flattened output of the second max pool. This is then passed through a single linear layer. 



| Layer No  | Functions     |Dimensions                                   |
|-----------|---------------|---------------------------------------------|
|Layer1:    |Conv           |kernel = 3x3, strides = 2 |
|           |Relu      |                                             |
|           |Max_Pool       |kernel = 3x3, strides = 2                  |
|Layer2:    |Conv           |kernel = 5x5, strides = 1 |
|           |Relu        |                                             |  
|           |Max_Pool       |kernel = 2x2, strides = 3 |
|Layer3:  |Conv           |kernel = 5x5, strides=2   |
|Flatten:    |Merge of Layer 3 and Layer 2 MaxPool |                    |    
|Layer4:    |Fully Connect  |

For parameter tuning the Adam optimizer was used to automatically adjust the learning rate. Epoches were adjusted such that the Validation Loss was near it's lowest value, at 5 epochs (see Model Fitting below). 

Softmax activation functions were tested instead of Relu's, usng various epochs (5,10,15) and testing with and without dropouts, this generally resulting in underfitting and poor autonomous driving. 

#### 2. Data Collection, Preprocessing, and Training Strategy

The original data used for training and validation included two passes around the track (counter-clockwise) of straight line driving and one pass of straight line driving around the track in the opposite directon (clockwise). This resulted in autonomous driving that tended veer off the track on longer, tighter curves and at key locations (such as bridges and the "dirt pull-off"). To combat this, additional data was collected, 

  1) that continously veered toward the curbs and "jerked" back to the center of the track, and  
  2) that repeatedly veered toward those key locations (bridge, dirt pull-off) and "jerked" back to the center. This helped train the model to avoid driving off of and into these key locations. However, the car tended to drive up onto and along the curb. 
  
Finally, data was collected driving along the second track, one time around, in both directions. This corrected the curb-driving behavior.

For data preprocessing normalization was used via the Lamdba() function using the equation (img/255 - 0.5) as suggested in lecture. Images were cropped by 50 rows from the top and 20 rows from the bottom, also suggested in lecture. Additional preprocessing methods were tested (converting to gray-scale and Canny edge detection) however this methods did not improve autonomous driving. 


#### 2. Model Fitting

At about 5 epochs the training loss and validation loss were relatively closs, with a training to validation loss ratio of 0.448 (where 1.0 would be equal, and a ratio > 1.0 would indicate underfitting the model). This keep the car on the track. The model was trained using dropouts after the second layer, after the third layer (immediately before the linear layer), and after both the second and third layers using keep probablities of 0.8 and 0.5, Epochs with the various dropout combinations were tested at 5,7,10,15. These did not appreciably improve overfitting, and autonomous driving was worse (the car always drove off of the track). Therefore dropouts were not used in the final model. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).





#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
