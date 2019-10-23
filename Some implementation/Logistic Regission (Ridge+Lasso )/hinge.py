from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regression constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


	loss = maximum(np.zeros((xTr.shape[1],1)),1 - np.matmul(xTr.T,w) * yTr.T).sum(axis=0) + lambdaa * np.matmul(w.T,w)
	?loss = np.squeeze(loss)		
	#print(loss)
	d = np.where(np.matmul(xTr.T,w) * yTr.T < 1 , 1, 0)
	d = np.diag(np.squeeze(d)) 
	gradient = - np.squeeze(np.matmul(np.transpose(yTr.T * xTr.T),d).sum(axis=1)).reshape((w.shape[0],1)) + 2 * lambdaa * w

	#print(np.squeeze(np.matmul(np.transpose(np.transpose(yTr)*np.transpose(xTr)),d).sum(axis =1)).reshape((1024,1))+w)
	#print(w.shape)

	return loss,gradient
