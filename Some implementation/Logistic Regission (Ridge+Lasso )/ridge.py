
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);
   
    loss = np.matmul(np.transpose(np.matmul(xTr.T,w)-yTr.T),(np.matmul(xTr.T,w)-yTr.T)) + lambdaa*np.matmul(w.T,w)
    # loss = np.squeeze(loss)
    gradient = 2*np.matmul(xTr,np.matmul(xTr.T,w)-yTr.T) + 2*lambdaa* w 
    
    return loss,gradient