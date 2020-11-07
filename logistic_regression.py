import numpy as np
import math
import matplotlib.pyplot as plt

data = np.loadtxt("/home/bhanu/Desktop/logic_data_1.txt",delimiter = ',')

x1 = data[:,0]

x2 = data[:,1]

y  = data[:,2]

#print(data)
#print(x2)
#print(x1)

m1	= 0.5
m2	= 0.5
c	= 0.3
alpha	= 0.1
iterations = 1000

x1_mean	= sum(x1)/len(x1)

x1_max	= max(x1)
x1_min	= min(x1)



x2_mean	= sum(x2) / len(x2)

x2_max	= max(x2)
x2_min	= min(x2)

x1_mean_data	= (x1 - x1_mean) / (x1_max - x1_min)
x2_mean_data	= (x2 - x2_mean) / (x2_max - x2_min)

print("x1_mean values\n:")
print(x1_mean_data)	

print("\nx2_mean values:\n")
print(x2_mean_data)

loss=[]

for i in range(iterations):

	z	= m1*x1_mean_data + m2*x2_mean_data + c
	
	exp	= np.exp(-z)
	
	m	= 1+exp

	y_pred	= 1. / m

	grad_x1	= x1_mean_data*(y_pred - y)

	grad_x1	=(1./len(grad_x1)) * sum(grad_x1)	


	grad_x2 = x2_mean_data*(y_pred -  y)

	grad_x2 = (1./len(grad_x2)) * sum(grad_x2)


	grad_c	= y_pred -  y

	grad_c	= 1./len(grad_c) * sum(grad_c)

	m1	= m1 - alpha * grad_x1
	
	m2	= m2 - alpha * grad_x2
	
	c	= c  - alpha * grad_c
	
	log_1	= y*np.log(y_pred)

	log_0	= (1-y)*np.log(1-y_pred)

	loss.append( (-1/len(y)) * sum(log_1 + log_0))




iter_list = range(0,iterations)
#plt.plot(iter_list,loss)
#plt.show()


pred_y = m1*x1_mean_data + m2*x2_mean_data + c

classes =[1 if logit>0.5 else 0 for logit in pred_y]

slope = -m2/m1

#plt.plot(x1_mean_data,x1*slope)
#plt.hold(True)
plt.scatter(x1_mean_data,x2_mean_data,c=classes)

plt.show()

diff = classes - y

accuracy = 1.0 - (float(np.count_nonzero(diff)) / len(diff))*1.0


print(m1,m2,c)





