# **Traffic Sign Recognition** 
---
**Building a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the [data set](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./fromweb/General_caution.jpg "Traffic Sign 1"
[image5]: ./fromweb/road_work.jpg "Traffic Sign 2"
[image6]: ./fromweb/speed_breaker.jpg "Traffic Sign 3"
[image7]: ./fromweb/stop.jpg "Traffic Sign 4"
[image8]: ./fromweb/yield.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Let's begin

Here is a link to my [project code](https://github.com/PraveenKumar-Rajendran/CarND-Traffic-Sign-Classifier-Project/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Basic summary of the data set. In the code, the analysis done using python, numpy.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed for the classes available in the whole dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Pre-process the Data Set (normalization, grayscale, etc.)

As a first step, I decided to normalize the images to using `cv2.normalize` method from openCV

Later, normalized RGB channel image is converted to grayscale before it is used for training.

Preprocessed image size will be `(32,32,1)`

<!-- ![alt text][image2] -->

<!--
As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

-->

#### 2. Final model architecture ( model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 2x2 stride, same padding, outputs 32x32x6 	|
| tanh activation		|												|
| Average pooling	   	| 2x2 stride, valid padding, outputs 16x16x6	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 12x12x6  	|
| tanh activation		|												|
| Flatten layer 	   	| output shape 576 1D                          	|
| Dense layer1 (120)   	| output shape 120 1D      						|
| tanh activation		|												|
| Dense layer2 (84)   	| output shape 84 1D      						|
| tanh activation		|												|
| Dense layer3 (43)   	| output shape 43 1D      						|
| Softmax				|           									|
 
`Total params: 85,631`

`Trainable params: 85,631`

`Non-trainable params: 0`


#### 3. Training the deep learning model.

To train the model, I used `adam optimizer` with `categorical_crossentropy` loss function. Model is trained for `20` epochs.

Tensorboard callback is included in training for the visualization of the model architecture and the training progress.

Preprocessed validation set is included in the model training progress.

#### 4. The discussed model architecture gave the training accuracy more than 99% and a validation accuracy more than 93% which is fair enough to do the given task.

My final model results were:
* training set accuracy of `0.9959`
* validation set accuracy of `0.9311`
* test set accuracy of `0.9244`

<!-- 
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

-->

Modified Lenet architecture was chosen and implemented in TensorFlow 2.x
As you can refer from the above given details this architecture did a pretty good job at validation set as well as the test set which can be a proof to believe it will perform fairly on the unseen data given that same preprocessing steps where used before the prediction process.
 
### Test a Model on New Images

#### 1. following five German traffic signs found on the web is gonna be used for the prediction.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

<!--
The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

-->

#### 2. Results summary

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General_caution      	| General_caution   							| 
| road_work     		| road_work 							    	|
| Bumpy Road			| Bumpy Road									|
| stop	        		| stop					 			        	|
| yield		        	| yield      						        	|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of `92.44%`

#### 3. Top 5 probabilities for the prediction on the single image.

The code for making prediction with top 5 probabilities on my final model is located in the last but one code cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a General caution sign (probability of 0.9), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.9999321e-01  		| General caution   							| 
| 6.2955587e-06     	| Traffic signals								|
| 3.3307344e-07			| Pedestrians									|
| 4.8186561e-08	      	| Wild animals crossing	 			        	|
| 3.7624989e-08		    | Right-of-way at the next intersection        	|


#### Room for improvement.

- The deep learning model can be fed with even more no of data samples for training so that the model can generalize well on any country's traffic signal sign data.

- Validation set can be added with different distribution so that validation accuracy will be a good parameter to judge the performance on the unseen test dataset.

- overfitting can be overcame by using even lighter model architecture.


<!-- 

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

-->

