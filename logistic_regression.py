#importing required libraries 

import numpy as np
import math
import matplotlib.pyplot as plt


#loading the input data using numpy

data = np.loadtxt("/home/bhanu/Desktop/codes/logic_data_1.txt",delimiter = ',')

print(data)


#splitting the data

x1 = data[:,0]

x2 = data[:,1]

y  = data[:,2]


#Training Parameters 

m1	= 0.5		#assuming weight-m1

m2	= 0.5		#assuming weight-m2

c	= 0.3		#assuming bias - c

alpha	= 0.1		#assuming learning rate(alpha)

iterations = 1000	#Taking number of iterations to train the model


#input data contains larger values so to get better results
#Normalizing the input data to get better output by using "mean normalization" method

x1_mean	= sum(x1)/len(x1)

x1_max	= max(x1)

x1_min	= min(x1)



x2_mean	= sum(x2) / len(x2)

x2_max	= max(x2)

x2_min	= min(x2)

#mean normalization of x1 and x2 data
x1_mean_data	= (x1 - x1_mean) / (x1_max - x1_min)

x2_mean_data	= (x2 - x2_mean) / (x2_max - x2_min)



loss=[]

#looping through the entire dataset to train the model

for i in range(iterations):

	
	#finding y_predicted value using sigmoid activation function

	z	= m1*x1_mean_data + m2*x2_mean_data + c
	
	exp	= np.exp(-z)
	
	m	= 1+exp

	y_pred	= 1. / m
	

	#finding x1, x2, c gradients

	grad_x1	= x1_mean_data*(y_pred - y)

	grad_x1	=(1./len(grad_x1)) * sum(grad_x1)	


	grad_x2 = x2_mean_data*(y_pred -  y)

	grad_x2 = (1./len(grad_x2)) * sum(grad_x2)


	grad_c	= y_pred -  y

	grad_c	= 1./len(grad_c) * sum(grad_c)

	#updating the weights m1,m2
	
	m1	= m1 - alpha * grad_x1
	
	m2	= m2 - alpha * grad_x2
	
	#updating the bias value c
	c	= c  - alpha * grad_c
	
	log_1	= y*np.log(y_pred)

	log_0	= (1-y)*np.log(1-y_pred)

	loss.append( (-1/len(y)) * sum(log_1 + log_0))




iter_list = range(0,iterations)

pred_y = m1*x1_mean_data + m2*x2_mean_data + c

classes =[1 if logit>0.5 else 0 for logit in pred_y]

slope = -m2/m1


#scattering the data
plt.scatter(x1_mean_data,x2_mean_data,c=classes)

plt.show()

diff = classes - y

accuracy = 1.0 - (float(np.count_nonzero(diff)) / len(diff))*1.0

#printing the weights and bias values
print(m1,m2,c)





