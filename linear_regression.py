#importing required libraries

import numpy as np
import matplotlib.pyplot as plt


#loading data from path /home/bhanu/Desktop/machine learning/linear_regression/simple_linear_data.txt 


data = np.loadtxt("/home/bhanu/Desktop/machine learning/linear_regression/simple_linear_data.txt",delimiter = ',')



#taking the input data from the dataset into given variables
#considering the coloumn-0 values into x variable


x = data[:,0]



#considering the coloumn-1 values into y_variable 


y = data[:,1]



#training parameters
#initializing weights and bias values to 0


m = 0

c = 0

learning_rate = 0.01		#learning rate (alpha)

no_of_iter = 1000		#number of iterations to train 



#looping by the number of iterations

for i in range(no_of_iter):
	

		
	#linear hypothesys

	y_pred = m*x + c


	#finding out the weight gradients (i.e, gradients of m)

	grad_m = 2.*x*(y_pred-y)

	grad_m = 1./len(grad_m)*sum(grad_m)



	#finding out the bias gradients (i.e gradients of c)
	
	grad_c = 2.*(y_pred - y)
		
	grad_c = 1./len(grad_c)*sum(grad_c)

		

	#updating the weights

	m = m-(learning_rate*grad_m)
		
		
	#updating the  biases

	c = c-(learning_rate*grad_c)




#printing the grad_m and grad_c

print(grad_m, grad_c)

#finding the predicted y with respect to the updated weight, bias values 

y_pred = m*x + c


#scattering the data of x and y

plt.scatter(x,y)

#plotting between x and predicted y (i.e y_pred)  to check how much our model is correct
#where r = red color

plt.plot(x,y_pred,color = 'r')
plt.show()

