#importing librarires

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
 

#importing cifar-10 data from tensorflow.keras.datasets

from tensorflow.keras.datasets import cifar10

#cifar dataset parameters

num_features	= 3072	#image featuers 32*32*3 = 3072 

num_classes	= 10	#total classes 0-9

#training parameters

batch_size	= 128	#batch size for each iteration

display_step	= 1	#display step to print the loss and accuracy with respect to the display step

iterations	= 2	#number of iterations

learning_rate	= 0.01	#learning rate alpha


#network parameters

conv_layer1_filters	= 32	#convolution layer1 filters

conv_layer2_filters	= 64	#convolution layer2 filters

conv_layer3_filters	= 128	#convolution layer3 filters

conv_layer4_filters	= 256	#convolution layer4 filters

fc1_units		= 256	#fully connected layer units

drop_out		= 0.25	#drop out to prevent the network from overfitting with rate of 0.25



#loading data into the given variables variables

(x_train, y_train), (x_test, y_test)	= cifar10.load_data()

#converting x_train and x_test datasets into float dataset values

x_train, x_test	= np.array(x_train,np.float32), np.array(x_test, np.float32)

#normalizing the x_train and x_test dataset

x_train, x_test	= x_train/255. ,x_test/255.



# Use tf.data API to shuffle and batch data


train_data	= tf.data.Dataset.from_tensor_slices((x_train,y_train))

train_data	= train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)



#random value generator to initialize weights

random_normal	= tf.initializers.RandomNormal()



weights		= {

		  
		  #convolution layer_1: 5x5 conv, 3 input, 32 filters
		  #output shape (5,5,3,32)

		 'wc1': tf.Variable(random_normal([5,5,3,conv_layer1_filters])),
		  
		  #convolution layer_2: 3x3 conv, 32 inputs, 64 filters
		  #output shape (3,3,32,64)

		  'wc2': tf.Variable(random_normal([3,3,conv_layer1_filters,conv_layer2_filters])),
		  
		  #convolution layer_3: 3x3 conv, 64 input, 128 filters
		  #output shape (3,3,64,128)

		  'wc3': tf.Variable(random_normal([3,3,conv_layer2_filters,conv_layer3_filters])),
		
		  #convolution layer_3: 3x3 conv, 128 input, 256 filters
		  #output shape (3,3,128,256)

		  'wc4': tf.Variable(random_normal([3,3,conv_layer3_filters,conv_layer4_filters])),
		
		  #fully connected layer_1: 5*5*256 inputs, 256 units
		  #output shape (5x5x256, 256)
		  'wfc1': tf.Variable(random_normal([5*5*256,fc1_units])),
		
		  #fully connected output: 256 inputs, 10 outputs
		  #output shape (256, 10)

		  'wout': tf.Variable(random_normal([fc1_units, num_classes]))}



#initiliazing bias values 

biases		= {

		  
		  #convolution layer1 bias values(32)
		  #output shape (32, )

		  'bc1': tf.Variable(tf.zeros([32])),

		  #convolution layer2 bias values(64)
		  #output shape (64, )

		  'bc2': tf.Variable(tf.zeros([64])),
		 
		  #convolution layer3 bias values(128)
		  #output shape (128, )

		  'bc3': tf.Variable(tf.zeros([128])),

		  #convolution layer4 bias values(256)
		  #output shape (256, )

		  'bc4': tf.Variable(tf.zeros([256])),
	
		  #fully connected layer1 bias values(32)
		  #output shape (256, )

		  'bfc1': tf.Variable(tf.zeros([256])),

		  #fully connected output bias values(32)
		  #output shape (10, )

		  'bout': tf.Variable(tf.zeros([10])) }




#create the network model

def conv_net_model(x, strides = 1, k = 2):


	
	# Input shape: [-1, 32, 32, 3]. A batch of 32x32x3 (RGB) images.
	x	= tf.reshape(x, [-1,32,32,3])


	#convolutional layer1 with input shape (32,32,3)
	#height = width = 32, channels = 3
	#output shape (128,28,28,32)
	#height = width = 28 channels = 32
	conv_layer1	=	tf.nn.conv2d(x, weights['wc1'],strides = [1,strides,strides,1],padding = 'VALID')

	conv_layer1	=	tf.nn.bias_add(conv_layer1, biases['bc1'])

	conv_layer1	=	tf.nn.relu(conv_layer1)

	#convolutional layer2 with input shape (128,28,28,32)
	#height = width = 28 channels = 32
	#output shape (128,26,26,64)
	#height = width = 26 channels = 64	

	conv_layer2	=	tf.nn.conv2d(conv_layer1, weights['wc2'], strides = [1,strides,strides,1], padding = 'VALID' )

	conv_layer2	=	tf.nn.bias_add(conv_layer2, biases['bc2'])

	conv_layer2	=	tf.nn.relu(conv_layer2)

	
	#convolutional layer3 with input shape (128,26,26,64)
	#height = width = 26 channels = 64
	#output shape (128,24,24,128)
	#height = width = 24 channels = 128


	conv_layer3	=	tf.nn.conv2d(conv_layer2, weights['wc3'], strides = [1,strides,strides,1], padding = 'VALID')

	conv_layer3	=	tf.nn.bias_add(conv_layer3, biases['bc3'])
	
	conv_layer3	=	tf.nn.relu(conv_layer3)

	#convolutional layer3 after max_pooling with input shape (128,24,24,128)
	#height = width = 24 channels = 128
	#output shape (128,12,12,128)
	#height = width = 12 channels = 128

	conv_layer3	=	tf.nn.max_pool(conv_layer3, ksize = [1,k,k,1],strides = [1,k,k,1], padding = 'VALID')

	
	#convolutional layer4 with input shape (128,12,12,128)
	#height = width = 12 channels = 128
	#output shape (128,10,10,256)
	#height = width = 10 channels = 256

	conv_layer4	=	tf.nn.conv2d(conv_layer3, weights['wc4'], strides = [1,strides,strides,1], padding = 'VALID')

	conv_layer4	=	tf.nn.bias_add(conv_layer4, biases['bc4'])

	conv_layer4	=	tf.nn.relu(conv_layer4) 

	#convolutional layer4 with input shape (128,10,10,256)
	#height = width = 10 channels = 256 
	#after max_pooling ouput shape(128,5,5,256)
	#height = width = 5 channels = 256 
	
	conv_layer4	= 	tf.nn.max_pool(conv_layer4, ksize = [1,k,k,1], strides = [1,k,k,1], padding = 'VALID')

	
	#fc_layer1 converted into 1 dimensional vector with input vector (128,5,5,256)
	#fc_layer1 output shape after reshape (128,6400)

	fc_layer1	=	tf.reshape(conv_layer4, [-1,weights['wfc1'].get_shape().as_list()[0]])

	#fc_layer1 with input shape (128,6400)
	#output shape(128,256)
	fc_layer1	=	tf.add(tf.matmul(fc_layer1, weights['wfc1']), biases['bfc1'])

	fc_layer1	=	tf.nn.relu(fc_layer1)
	
	#drop out to prevent the network from overfitting

	fc_layer1	=	tf.nn.dropout(fc_layer1, drop_out)

	#fc out_layer with input shape (128,256)
	#output shape(128,10)
	
	out_layer	=	tf.add(tf.matmul(fc_layer1, weights['wout']), biases['bout'])

	
 	#Apply softmax to normalize the logits to a probability distribution.

	return tf.nn.softmax(out_layer)



#cross entropy loss function
def cross_entropy(y_pred, y):


	
	#encode all the labes  to a one hot vector

	y	= tf.one_hot(y, depth = num_classes)
	
	#clip prediction values to avoid log(0) error	
	#input shape (128,10)
	#ouput shape(128,10)

	y_pred	= tf.clip_by_value(y_pred, 1e-9,1.)
	
	#compute and returning cross entropy loss

	return tf.reduce_mean(-(y * tf.math.log(y_pred)))


#accuracy metric
def accuracy(y_pred, y):
	
	
	#predicted class is the index of highest score in prediction vector (i.e. argmax)
	
	correct_prediction	= tf.equal(tf.argmax(y_pred, 1),tf.cast(y, tf.int64))
	
	return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	

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


#run training for the given number of steps 
for step, (batch_x, batch_y) in enumerate(train_data.take(iterations),1):	

	
	#run optimization to update the w and b values
	
	run_optimization(batch_x, batch_y)

	
	#if step % display_step then update the pred, loss and accuracy values
	if step % display_step == 0:

		pred	= conv_net_model(batch_x)

		loss	= cross_entropy(pred, batch_y)

		acc	= accuracy(pred,batch_y)
		
		#printing the step number and the respective loss and accuracy values
		print(loss, acc)



#test model on validation set 

pred  = conv_net_model(x_test)

print("Testing accuracy:%f"%(accuracy(pred, y_test)))



#predicting the first 3 images of the x_test data set
predictions = conv_net_model(x_test[:3])

#looping through testing dataset by range 4
for i in range(2):

    plt.imshow(np.reshape(x_test[i],[32,32,3]))

    plt.show()
