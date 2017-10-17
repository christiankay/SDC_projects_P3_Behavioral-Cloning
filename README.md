


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
[//]: # (Image References)

[image1]: ./examples/test_normalized_3images_nvmodel_corr0.1_ep7.png "Model mean squared error loss"
[image2]: ./examples/center_2017_10_14_16_59_20_791.jpg "center driving"
[image3]: ./examples/center_2017_10_14_16_59_22_088.jpg "left lane line"
[image4]: ./examples/center_2017_10_14_16_59_22_360.jpg "Recovery Image"
[image5]: ./examples/center_2017_10_14_16_59_22_635.jpg "Recovery Image"
[image6]: ./examples/right_2017_10_14_17_33_13_765.jpg "Normal Image"
[image7]: ./examples/right_2017_10_14_17_33_13_765_flip.png "Flipped Image"

### Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


#### Files Submitted & Code Quality

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


---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 8x8, 5x5, 3x3 filter sizes and depths between 16 and 64 (model.py lines 96-104) 

The model includes ELU layers to introduce nonlinearity (code line 97-111) and due to its mean value of zero the activition function may provide faster learning rates. The data is also normalized in the model using a Keras lambda layer (code line 93). 

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 107 and 110). 

The model was trained and validated on different data sets (80% training data, 20% validation data) to ensure that the model was not overfitting (code line 24). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. After that, the model had problems to stay on track mainly at two different positions. The first position was the first curve before the bridge on track one. The car tended to go straight into the water. After gathering more images of this part of the track, the model was able to follow the track appropriate. The same process was applied for the first left corner after the bridge, where the road lanes on the right side are missing.

For details about how I created the training data, see the next section. 

---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a stack of convolutional layers with non-linear activation functions to extract relevant features from the image data. Fully connected layers were used at the and of the model to connect the convolutional layers with the steering angle / output. 

My first step was to use a convolution neural network model similar to the NVIDIA self driving car model. I thought this model might be appropriate because it has different convolutional layers in sequence connected via activation functions to extract complex geometries from the training images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it had less parameters (dropout layers).

Then I increased the training data set by using also flipped images(vertical) and inverted steering angles. I also created additional image data from the left and right camera of the simulator. In that case, the steering angle was slightly corrected by a constant factor (code lines 53 and 60). 

The model was built by training the model through 5 epochs. For now only models with this amount of epochs are able to successful drive on the track. I have also training models with 3 and 7 epoch both approaches are not feasible.

![alt text][image1]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 96-104) consisted of a convolution neural network with the following layers and layer sizes

1. convolutional layer: 16x8x8, ELU activation
2. convolutional layer: 16x8x8, ELU activation
3. convolutional layer: 32x5x5, ELU activation
4. convolutional layer: 64x5x5, ELU activation
5. convolutional layer: 32x3x3, ELU activation
6. dropout layer (drop 20% of input neurons)
7. ELU activation
8. Fully connected layer (512 output neurons)
9. dropout layer (drop 50% of input neurons)
10. ELU activation
11. Fully connected layer (1 output neuron)



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on center of the track. These images show what a recovery looks like starting from left lane line:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data set (see section above), I also flipped images and angles thinking that this would help to avoid overfiiting and decrease the influence of my personal driving behaviour. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had about 80,000 number of data points. I then preprocessed this data by removing the first 70 pixel lines from the top and the first 20 from bottom (code line 91)


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by testing different amounts of epochs. The lowest validation mean square error over all epochs was not a good indicator for a feasible model. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The model was trained on my local GPU (NVIDIA GTX 770) and also the online prediction of the steering angle was calculated by the GPU. For loading training data a python generator was used with a batch size of 128 data points to avoid memory problems. 
