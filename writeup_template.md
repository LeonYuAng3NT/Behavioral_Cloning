#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


[//]: # (Image References)

[image1]: ./images/steering_angle.png "Streering angle"
[image2]: ./images/training_set.png "training"
[image3]: ./images/recovery_Image.gif "Recovery"
[image4]: ./images/histogram.png "Histogram Image"



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Preprocess the data and data collection

I record driving data in the form of images and correlated steering wheel angles
for later training set.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and captured

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to first prepare the data into a pickle file and then retrieve data from it and let the neural network to learn from it.

My first step was to use a convolution neural network model similar to the Keras lambda layer and I thought this model might be appropriate because it is simple and straightforward

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that the dropout layers will
be used after each Convolutional Layer to avoid overfitting issue.

Then I tried to modified the batch size add multiple Dropout function to see the result. I printout the loss of each Epoch and notice that this approach will reduce the loss to less than 0.01 which is a standout performance compared to others.

An Adam Optimizer was used with SGD as the loss function during training, the final values for the loss function were around .024 for both the training and validation set with about 40,000 training examples and 400 validation. To test the model accuracy even further the second track was tested next. The test runs on both track are generally successfully except the fact that the vehicle runs in in track 1 tends to steer arbitrarily at first and does not stay in the centre,but it will eventually go back to the track. This problem is probably due to the human error made by my training sets because I couldn't control the vehicle with valid accuracy at the first time and I did record it. However it was actually very easy to train the car to drive all the way through track 2 after minimal training with its already configured weights.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes

____________________
Layer (type)                     Output Shape          Param #     Connected to                     
=========================================================================================
lambda_1 (Lambda)                (None, 40, 80, 3)     0   lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 40, 80, 16)    3088        lambda_1[0][0]                   
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 40, 80, 16)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 40, 32)    12832       elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 20, 40, 32)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 10, 20, 64)    51264       elu_2[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 12800)         0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 12800)         0           flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 12800)         0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           6554112     elu_3[0][0]                      
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 512)           0           dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1)             513         elu_4[0][0]                      
================================================================================

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


![alt text][image3]


After the collection process, I trained the program with my model and below are a few examples of the training set images and the corresponding steering angles s (in radians).

![alt text][image2]

I then added histograms and interpret that the steering angle in the entire dataset. s=0 have the highest occurence: more than 20 times the frequency of other angle values. Steering angles around the value of zero will be much more frequent that steeper turns, since roads are more often than not straight.Furthermore, there are more positive (1900 counts) than negative angle values (1775 counts). The frequency decreases with increasing steering angle value. For |s| > 0.5rad, the counts are negligible.


![alt text][image1]

Below is the Sample Data Distribution. The straight angle (zero degree) has far more chance feeding into model, where the real turn looks becomes very minor to system.It becomes evident that it will be necessary to balance this dataset prior its use. If we use the data without further processing, the model would predict new steering angles with a very strong bias towards going straight, which would become problematic when turning.The adopted solution may involve defining a range of steering ranges around 0 that will be sampled with reduced frequency compared to the rest of steering angle values.By careful selection of both the steering angles range and the frequency of sampling, the distribution of steering angle values will be much more suitable for training our network.

![alt text][image4]

I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
