#importing required libraries

import  tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt


#loading data using np.load() function from path /home/bhanu/Desktop/machine learning/linear_regression/simple_linear_data.txt 

data = np.loadtxt("/home/bhanu/Desktop/machine learning/linear_regression/simple_linear_data.txt",delimiter = ',')



#considering the data into given variables
#cosidering the coloumn-0 data into x variable


x	= data[:,0]


#considering the coloumn-1 data into y variable


y	= data[:,1]


#initializing the weights and bias values randomly using tensorflow tf.Variable()

w	=	tf.Variable(np.random.randn(),name = 'weight')

b	=	tf.Variable(np.random.randn(),name = 'bias')



#training parameters


iterations	= 1100		#number of iterations to train model

learning_rate	= 0.01		#learning rate to learn the model

batch_size	= 20		#batch size for each iteration 
	


def linear_regression(x):
	
	#returning the linear hypothesis

	return w*x + b



#considering mean square loss taking inputs predicted y and y_true value(i.e ground truth y)
def mean_square(y_pred,y):

		
	#returning mean square loss	

	return tf.reduce_mean(tf.square(y_pred - y))


#taking stochastic gradient descent (SGD) optimizer 

optimizer	= tf.optimizers.SGD(learning_rate)

#running the  optimization 
def run_optimization():


	#wrap computation inside a GradientTape for automatic differentiation
	with tf.GradientTape() as g:
	
		#calling the linear regression function
		pred	= linear_regression(x)		
		
		#calling the loss function
		loss	= mean_square(pred,y)
		
		#computing the gradients
		gradients = g.gradient(loss,[w,b])
	
	#updating the gradients  
	optimizer.apply_gradients(zip(gradients,[w,b]))


#looping by the number of iterations
for step in range(iterations):

	
	#calling functon run optimization
	run_optimization()


	#if step % batch_size == 0 then call the functions linear_regression and mean_square
	if step % batch_size	== 0:
	
		pred	= linear_regression(x)
		loss	= mean_square(pred,y)


#printing the loss function

print(loss)

	
#finding the predicted_y  with updated gradients (i.e w and b)

y_pred	= np.array (w * x + b)


#scattering the x and y values on a 2 dimensional plot
plt.scatter(x,y)

#plotting between the x and predicted_y to check how much model is correct

plt.plot(x,y_pred,color = 'r')
plt.show()

