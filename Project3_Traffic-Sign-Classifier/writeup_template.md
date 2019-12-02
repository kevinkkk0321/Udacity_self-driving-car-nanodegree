# **Traffic Sign Recognition** 

## Writeup Template

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image_record/image1.png 
[image2]: ./image_record/image2.png 
[image3]: ./image_record/image3.png 
[image4]: ./image_record/image4.png 
[image5]: ./image_record/image5.png 
[image6]: ./image_record/image6.png 

[image7]: ./web_images/images.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kevinkkk08/Udacity_self-driving-car-nanodegree/blob/master/Project3_Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...
#origianal data
![alt text][image1]
#augmented data
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In order to augment my images, and add more data to the the data set, I used the following techniques:  
Resize: 50% / Rotate -30~30 degree / Warp 

Here is an example of an original image and an augmented image:

#origianal image
![alt text][image3]
#augmented image
![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16				|
| Flatten		| 5x5x16 -> 400						   |
| Dropout		| keep_prob = 0.8						 |
| Fully connected		| 400 -> 120				|
| RELU					|				
| Dropout		| keep_prob = 0.75					 |
| Fully connected		| 120 -> 84    	|
| RELU					|				
| Fully connected		| 84 -> 43						|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I use the hyperparameters as follow:

| hyperparameter			        |     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| EPOCHS		        | 80				       |
| BATCH_SIZE					 |	128			       |
| learning_rate		 | 0.0006	      |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

| Set			        |     accuracy	        					| 
|:---------------------:|:---------------------------------------------:| 
| Training set		      | 0.989				       |
| Validation set					 |	0.940			        |
| Test	set          	 | 0.915           |


If a well known architecture was chosen:
* What architecture was chosen? LeNet

First, implement regularization to regularize the images. 

Second, tune the learning rate lower (0.0006) and add more epochs (80) for higher accuracy rate. 

Third, base on the LeNet model, I added two dropout layer in it in order to avoid overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No_entry      		| No_entry   									| 
| Road_work     			| Road_work 										|
| Speed limit (30km/h)				| Yield											|
| Stop	      		| Stop				 				|
| Turn_right_ahead			| Roundabout mandatory      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 
I think it's the image I collect from web(which are big) were distorted after resize, causing the predict accuracy only got 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .00        | Speed limit (60km/h)   									| 
| .00     			| Speed limit (50/h) 										|
| .00					   | Speed limit (30km/h)											|
| .00	      	| Speed limit (20km/h)		 				|
| 1.0 			    | No entry     							|

For the second image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| .00     			| Speed limit (50/h) 										|
| .00					   | Speed limit (30km/h)											|
| .00	      	| Speed limit (20km/h)		 				|
| .00        | Bicycles crossing   									| 
| 1.0 			    | Road work     							|

For the third one,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:|
| .00	      	| Speed limit (20km/h)		 				|
| .00        | Speed limit (80km/h)   									| 
| .00     			| Priority road     									|
| .00					   | Speed limit (30km/h)											|
| 1.0 			    | Stop     							|


For the fourth,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .00        | Speed limit (60km/h)   									| 
| .00     			| Speed limit (50/h) 										|
| .00					   | Speed limit (30km/h)											|
| .00	      	| Speed limit (20km/h)		 				|
| 1.0 			    | No entry     							|


For the fifth,

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .00     			| Speed limit (50/h) 										|
| .00					   | Speed limit (30km/h)											|
| .00	      	| Speed limit (20km/h)		 				|
| .00        | Turn right ahead   									| 
| 1.0 			    | Keep right     							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


