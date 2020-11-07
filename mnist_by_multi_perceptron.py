#importing required libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#mnist parameters
num_features	= 784		#num features 28x28

num_classes	= 10		#num classes 10

#training parameters
batch_size	= 256		#batch size for each size

display_step	= 50		#display step to print loss and accuracy with respect to the display step

iterations	= 2000		#number of iterations 
	
learning_rate	= 0.01		#learning rate alpha

#network parameters
n_hidden_1	= 128		#hidden layer1 units

n_hidden_2	= 256		#hidden layer2 units



#importing the mnist data from tensorflow.keras.datasets 
from tensorflow.keras.datasets import mnist
 

#loading mnist data into given variables

(x_train,y_train) , (x_test , y_test)	= mnist.load_data()

#converting the datasets into float32 values

x_train , x_test	= np.array(x_train,np.float32) , np.array(x_test,np.float32)

#converting the datasets into 1 dimensional vector

x_train , x_test	= x_train.reshape([-1 , num_features]) , x_test.reshape([-1 , num_features])

#normalizing the vector
x_train , x_test	= x_train / 255. , x_test / 255.



# Use tf.data API to shuffle and batch data

train_data	= tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data	= train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)


#random initializer to initialize the weights randomly

random_normal	= tf.initializers.RandomNormal()


weights	= {
		#hidden layer1 weights : shape(784,128)
		'h_1' : tf.Variable(random_normal([num_features,n_hidden_1])),
	
		#hidden layer2 weights : shape(128, 256)
		'h_2' : tf.Variable(random_normal([n_hidden_1 , n_hidden_2])),

		#out layer weights : shape(256,10)
		'out' : tf.Variable(random_normal([n_hidden_2 , num_classes]))
		
		}



#intializing biases values 

biases	= {
		#hidden layer1 bias values (128)
		'b_1' : tf.Variable(tf.zeros([n_hidden_1])),

		#hidden layer2 bias values (256)
		'b_2' : tf.Variable(tf.zeros([n_hidden_2])),
		
		#output layer bias values (10)
		'out' : tf.Variable(tf.zeros([num_classes]))
		
		}



#creating network model with input layer, 2-hidden layers, and output layer with 10 units

def neural_net(x) :
 

	#hidden layer1 with output shape (256,128)
 
	layer_1	= tf.add(tf.matmul(x, weights['h_1']), biases['b_1'])
	layer_1	= tf.nn.sigmoid(layer_1)


	#hidden layer2 with input shape (256,128)
	#output shape(256,256)

	layer_2	= tf.add(tf.matmul(layer_1 , weights['h_2']), biases['b_2'])
	layer_2 = tf.nn.sigmoid(layer_2)
	
	
	#ouput layer with input shape(256,256)
	#ouput shape(256,10)

	output_layer = tf.matmul(layer_2 , weights['out']) + biases['out']	
	
	
 	#Apply softmax to normalize the logits to a probability distribution.

	return tf.nn.softmax(output_layer)



#cross entropy loss function
def cross_entropy_loss(y_pred , y):

	
	#encoding all the y values into one hot vector
	
	y	= tf.one_hot(y , depth = num_classes)

	
	#clip prediction values to avoid log(0) error
	#input shape (256,10)
	#output shape (256,10)
	y_pred	= tf.clip_by_value(y_pred , 1e-9 , 1.)
	
	
	#compute and returning cross entropy loss
	return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred)))



#accuracy metric function
def accuracy(y_pred , y):


	#predicted class is the index of highest score in prediction vector (i.e. argmax)
	correct_prediction	= tf.equal(tf.argmax(y_pred , 1) , tf.cast(y, tf.int64))


	return tf.reduce_mean(tf.cast(correct_prediction , tf.float32) , axis = -1)


#Optimizer technique to train the model

optimizer = tf.optimizers.SGD(learning_rate)


#run optimization
def run_optimization(x,y) :

	
	#wrap computation inside a GradientTape for automatic differentiation
	with tf.GradientTape() as g :
	
		pred = neural_net(x)
		loss = cross_entropy_loss(pred , y)


	#variables to update i.e trainable variables

	trainable_variables	= list(weights.values()) + list(biases.values())

	#computing the gradients
	gradients  = g.gradient(loss , trainable_variables)

	#updating the gradients and trainable variables
	optimizer.apply_gradients(zip(gradients , trainable_variables))



#loopinf through the number of iterations to run the optimization process
for step, (batch_x , batch_y) in enumerate(train_data.take(iterations) , 1):

	#calling optimization function
	run_optimization(batch_x , batch_y)

	#if step % display_step == 0 then call and update pred, loss,acc
	if step % display_step == 0:	

		pred	= neural_net(batch_x)
		loss	= cross_entropy_loss(pred,batch_y)
		acc	= accuracy(pred , batch_y)

		#printing the step number and the respective loss and accuracy values
		print('\nstep: %i ,loss : %f , acc : %f ' %(step,loss,acc))


#testing on validation set

pred	= neural_net(x_test)

print("\ntesting accuracy : %f "%accuracy(pred,y_test))


#predicting first 5 images of the x_test dataset

n	=	5

test_images	= x_test[:n]

predictions	= neural_net(test_images)

#looping in range by n value 
for i in range(n):

	plt.imshow(np.reshape(test_images[i] , [28,28]) , cmap = 'gray')
	plt.show()
	print("model prediction : %i"%tf.argmax(predictions.numpy()[i]))

	
	
			

