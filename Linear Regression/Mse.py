import numpy as np 

def RMse(y,y1):
	# y, y1 are 1xn
	n = y.shape[0]
	mse = np.square(y-y1).sum()/n

	return(np.sqrt(mse))


