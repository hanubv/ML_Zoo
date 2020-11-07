#importing librarires

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

#importing mnist data from tensorflow.keras.datasets

from tensorflow.keras.datasets import mnist


#mnist dataset perameters

num_features	= 784	#image featuers 28*28 = 784 

num_classes	= 10	#total classes 0-9


#training parameters
	
iterations	= 1	#number of iterations

batch_size	= 128	#batch size for each iteration

display_step	= 1	#used to print loss and acuuracy with respect to the display step
	
learning_rate	= 0.001	#learning rate alpha


#network parameters

conv1_filters	= 32	#convolution layer1 filters

conv2_filters	= 64	#convolution layer2 features

fc_units	= 1024	#fully connected layer units

drop_out	= 0.25	#dropout



#loading data into respective variables

(x_train, y_train), (x_test, y_test)	= mnist.load_data()

#converting x_train and x_test datasets into float values

x_train, x_test 	= np.array(x_train, np.float32), np.array(x_test, np.float32)

#normalizing the x_train and x_test dataset

x_train,x_test	= x_train/255., x_test/255.


# Use tf.data API to shuffle and batch data

train_data	= tf.data.Dataset.from_tensor_slices((x_train,y_train))

train_data	= train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)



#random value generator to initialize weights 

random_normal	= tf.initializers.RandomNormal()


weights		=	{
			
			#convolution layer_1 weigths: 5x5 conv, 1 input, 32 filters
			#output shape (5,5,1,32)

			'wc1': tf.Variable(random_normal([5,5,1,conv1_filters])),
		
			#convolution layer_2 weights : 5x5 conv, 32 inputs, 64 filters 
			#output shape (5,5,32,64)

			'wc2': tf.Variable(random_normal([5,5,conv1_filters, conv2_filters])),
			
			#fully connected layer_1 weights: 7x7x64 input units, 1024 units
			#output shape (7x7x64, 1024)

			'wfc1': tf.Variable(random_normal([7*7*64, fc_units])),
		
			#fully connected output layer weights: 1024 input units, 10 output units
			#output shape (1024,10)

			'wout': tf.Variable(random_normal([fc_units,num_classes]))}
			


#initiliazing bias values 

biases		=	{

			#convolution layer1 bias values(32)
			#bc1 shape(32,)
	
			'bc1': tf.Variable(tf.zeros([conv1_filters])),
			
			#convolution layer2 bias values(64)
			#bc2 shape(64,)
	
			'bc2': tf.Variable(tf.zeros([conv2_filters])),
			
			#fully connected layer1 bias values(1024)
			#fc1 shape(1024,)

			'bfc1': tf.Variable(tf.zeros([fc_units])),
			
			#output layer bias values(10)
			#bout shape(10,)
	
			'bout': tf.Variable(tf.zeros([num_classes]))}
		



#create the network model

def conv_net_model(x, strides = 1, k = 2):

	
	
	#Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images. 
	#1 is the color channel and 28x28 is the image height and width

	x	= tf.reshape(x, [-1,28,28,1])


	#convolution layer_1 with input shape[28,28,1] 
	#height = width = 28, channels = 1
	#output shape [128,28,28,32]
	#height = width = 28, channels = 32

	conv_layer1	= tf.nn.conv2d(x, weights['wc1'], strides = [1,strides,strides,1], padding = 'SAME')
	
	#adding bias to the layer

	conv_layer1	= tf.nn.bias_add(conv_layer1,biases['bc1'])

	#applying relu activation function

	conv_layer1	= tf.nn.relu(conv_layer1)

	#maxpooling the convolution layer1 and output shape is [128,14,14,32]
	#hight = width = 14, channels = 32

	conv_layer1	= tf.nn.max_pool(conv_layer1, ksize = [1,k,k,1], strides = [1,k,k,1],padding = 'SAME')
	
	#convolution layer2 with input shape [128,14,14,32]
	#height = width = 14, channels = 32
	#ouput shape [128, 14, 14, 64]
	#hight = width = 14, channels = 64

	conv_layer2	= tf.nn.conv2d(conv_layer1,weights['wc2'], strides = [1,strides,strides,1], padding = 'SAME')

	#adding bias to convolution layer2

	conv_layer2	= tf.nn.bias_add(conv_layer2, biases['bc2'])
	
	#applying relu activation function

	conv_layer2	= tf.nn.relu(conv_layer2)

	#applying maxpooling to convolution layer2 and output shape is [128, 7, 7, 64]	

	conv_layer2	= tf.nn.max_pool(conv_layer2, ksize = [1,k,k,1], strides = [1,k,k,1], padding = 'SAME')

	#flattening convolution layer2 output image vector into 1-dimensional vector

	fc1	= tf.reshape(conv_layer2, [-1,weights['wfc1'].get_shape().as_list()[0]])
	
	#fully connected layer with input shape (7x7x64,1024) 
	#ouputshape [128,1024]
	
	fc1	= tf.add(tf.matmul(fc1,weights['wfc1']), biases['bfc1'])

	#applying relu activation function

	fc1	= tf.nn.relu(fc1)

	#applying dropout to fully connected layer to prevent overfitting
	fc1	= tf.nn.dropout(fc1, drop_out)

	#fully connected output layer output shape[128,10]
	out	= tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])

 	#Apply softmax to normalize the logits to a probability distribution.

	return tf.nn.softmax(out)


#cross entropy loss function
def cross_entropy(y_pred, y):

	
	#encode all the labes  to a one hot vector
	
	y	= tf.one_hot(y, depth = num_classes)
	
	#clip prediction values to avoid log(0) error	
	#input shape (128,10)
	#ouput shape(128,10)

	y_pred	= tf.clip_by_value(y_pred, 1e-9,1.)

	#compute and returning cross entropy loss

	return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred)))


#Accuracy metric
def accuracy(y_pred, y):

	
	#predicted class is the index of highest score in prediction vector (i.e. argmax)
	
	correct_prediction	= tf.equal(tf.argmax(y_pred, 1), tf.cast(y, tf.int64))
	
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis = -1)

#ADAM optimizer for training the model

optimizer	= tf.optimizers.Adam(learning_rate)


#optimization process

def run_optimization(x, y):

	
	#wrap computation inside a GradientTape for automatic differentiation
	
	with tf.GradientTape() as g:

			
		pred	= conv_net_model(x)
		
		loss	= cross_entropy(pred, y)

	#variables to update i.e trainable variables

	trainable_variables	= list(weights.values()) + list(biases.values())

	#computing gradients
	gradients	= g.gradient(loss, trainable_variables)

	#updating the w and b following gradients

	optimizer.apply_gradients(zip(gradients, trainable_variables))


#run the loop through the given number of iterations 
for step, (batch_x, batch_y) in enumerate(train_data.take(iterations),1):

	#run optimization to update the w and b values

	run_optimization(batch_x, batch_y)

	#if step % display_step then  update the pred, loss and accuracy functions
	if step % display_step == 0:

		pred	= conv_net_model(batch_x)
		
		loss	= cross_entropy(pred, batch_y)
	
		acc	= accuracy(pred, batch_y)

		#printing the step number and the respective loss and accuracy values
		print("\nstep: %i, loss: %f, accuracy: %f" %(step, loss, acc))


#test model on validation set 

pred	= conv_net_model(x_test)

print("Testing Accuracy: %f" %(accuracy(pred, y_test)))



#predicting the first 4 images of the x_test data set

predictions	= conv_net_model(x_test[:4])

#looping through testing dataset by range 4

for i in range(4):

	plt.imshow(np.reshape(x_test[i], [28,28]), cmap = 'gray')

	plt.show()

	print("model prediction: %i" %np.argmax(predictions.numpy()[i]))
