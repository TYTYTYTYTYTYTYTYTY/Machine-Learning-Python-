import math
import numpy as np

'''

	INPUT:
	xTr dxn matrix (each column is an input vector)
	yTr 1xn matrix (each entry is a label)
	w weight vector (default w=0)

	OUTPUTS:

	loss = the total loss obtained with w on xTr and yTr
	gradient = the gradient at w

	[d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):
	loss = np.log(1+np.exp( - yTr.T* np.matmul(xTr.T,w))).sum(axis=0)
	
	#if yTr.shape[1]>1:
	gradient = ((-xTr.T*yTr.T*(1/(1+np.exp(np.matmul(xTr.T,w)*yTr.T))))).sum(axis=0).reshape((w.shape[0],1))
	
	#else : 
		#gradient = ((-np.transpose(xTr)*np.transpose(yTr)*(1/(1+np.exp(np.matmul(np.transpose(xTr),w)*np.transpose(yTr)))))).reshape((w.shape[0],1))

	return loss,gradient
