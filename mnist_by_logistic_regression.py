#importing the required libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#MNIST dataset perameters
num_classes	= 10		#number of classes
	
num_features	= 784		#number of features 28x28

#computation requirements

iterations	= 1010		#number of iterations

batch_size	= 256		#batch size for each iteration

display_step	= 50		#display step to print loss and accuracy with respect to the display step

learning_rate	= 0.01		#learning rate


#loadning data from mnist dataset from tensorflow.keras.datasets

from tensorflow.keras.datasets import mnist

#considering training data and testing data into given variables

(x_train,y_train) , (x_test,y_test)	= mnist.load_data()

#converting x_train and x_test datasets into float32 values

x_train , x_test	=	np.array(x_train,np.float32) , np.array(x_test,np.float32)

#reshaping the  training and testing vectors into 1 dimensional vectors
		
x_train , x_test	=	x_train.reshape([-1,num_features]) , x_test.reshape([-1,num_features])

#normalizing the x_train and x_test vectors

x_train , x_test	= x_train/255. , x_test/255.


# Use tf.data API to shuffle and batch data

train_data	=	tf.data.Dataset.from_tensor_slices((x_train,y_train))	
train_data	=	train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)



#initilizing the weights and bias

w	=	tf.Variable(tf.ones([num_features,num_classes]),name = 'weight')
b	=	tf.Variable(tf.zeros([num_classes]),name	= 'bias')



#multiclass logistic funtion
def logistic_regression(x):

 	#Apply softmax to normalize the logits to a probability distribution.
	return tf.nn.softmax(tf.matmul(x,w) + b)



#multiclass cross-entropy loss
def cross_entropy(y_pred,y):

	#ecoding the y values into one hot encoder values	
	y	= tf.one_hot(y , depth = num_classes)
	
	#clip prediction values to avoid log(0) error
	y_pred	= tf.clip_by_value(y_pred , 1e-9 , 1.)

	return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred),1))


#accuracy metric
def accuracy(y_pred,y):
	
	
	#predicted class is the index of highest score in prediction vector (i.e. argmax)
	correct_prediction = tf.equal(tf.argmax(y_pred,1) , tf.cast(y,tf.int64))

	return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#stochastic gradient descent optimization technique for better optimizing process

optimizer	= tf.optimizers.SGD(learning_rate)


#running optimization

def run_optimization(x,y):
	

	#wrap computations inside a gradient tape for automatic differentiation
	with tf.GradientTape() as g:

		pred	= logistic_regression(x)
		loss	= cross_entropy(pred,y)

	#computing gradients loss,w and b
	gradients = g.gradient(loss,[w,b])

	#updating the weights and bias gradients
	optimizer.apply_gradients(zip(gradients,[w,b]))



#looping through the batch with number of iterations

for step , (batch_x,batch_y)  in enumerate(train_data.take(iterations),1):


	#run optimization to update the graidents
		
	run_optimization(batch_x,batch_y)
	 
	
	#if step % display_step then update the pred, loss and accuracy values
	if step % display_step == 0:

		pred	= logistic_regression(batch_x)
		loss	= cross_entropy(pred,batch_y)
		acc	= accuracy(pred,batch_y)

		print("\naccuracy = ",acc)


#test model on validation set

pred = logistic_regression(x_test)

print("Test Accuracy: %f" % accuracy(pred, y_test))


#predicting the first 5 images of the x_test data set
n = 5

test_images = x_test[:n]
predictions = logistic_regression(test_images)


#looping through testing dataset by range 5
for i in range(n):


	plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
	plt.show()
	print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))

	
