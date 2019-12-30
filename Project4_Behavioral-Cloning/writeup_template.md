# **Behavioral Cloning** 

## Writeup Template

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./train_images/Fig1.jpg "center"
[image2]: ./train_images/Fig2.jpg "right"
[image3]: ./train_images/Fig3.jpg "mid"
[image4]: ./train_images/Fig4.jpg "center"
[image5]: ./train_images/Fig5.jpg "flipped"
[image6]: ./train_images/Fig6.jpg "cropped"
[image7]: ./train_images/Fig7.jpg "special_turn"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 2x2 filter sizes and depths between 24 and 64 (model.py lines 101-106).

The model includes RELU layers to introduce nonlinearity (code lines 101-106), and the data is normalized in the model using a Keras lambda layer (code line 100). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. (model.py line 102)

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 113).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model architecture I used was inspired from a similar network employed by NVIDIA team for steering control of an autonomous vehicle. 

The final goal was to run the simulator to see how well the car was driving around the track. There were a few spots where the vehicle fell off the track. In order to improve the driving behavior in these cases, I augmented the training data of the specific turns and train it. And the car finally stays on the track! 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture consisted of 4 convolution neural network as following architecture.

Here is a visualization of the architecture


| Layer                            |    Size       |
| --------------------             |:-------------:|
| Input                            | 160 x 320 x 3  |
| Lambda (normalization)           | 80 x 320 x 3  |
| Convolution with relu activation | 5 x 5 x 24 with 2x2 filters  |
| Dropout                          | 0.5  |
| Convolution with relu activation | 5 x 5 x 36 with 2x2 filters  |
| Convolution with relu activation | 5 x 5 x 48 with 2x2 filters  |
| Convolution with relu activation | 3 x 3 x 64   |
| Convolution with relu activation | 3 x 3 x 64   |
| Flatten                          |              |
| Fully connected                  | 100          |
| Fully connected                  | 50          |
| Fully connected                  | 10          |
| Output                           | 1          |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help generalize the model. For example, here is an image that has then been flipped:

![center][image1]  ![alt text][image5]

In my training process, my car always drive out the lane in a specific turn, so I add more training images of that turn, as below:

![alt text][image7]

After the collection process, I had 32187 number of data points. I then preprocessed this data by cropping irrelevant data from the top of the image. This led to a final image size of 80 x 320 x 3.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 since the loss didn't decrease much after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
