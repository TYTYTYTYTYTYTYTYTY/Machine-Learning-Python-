import numpy as np
from spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from valsplit import valsplit
from scipy import io
from checkgradHingeAndRidge import checkgradHingeAndRidge
from checkgradLogistic import checkgradLogistic
from ridge import ridge
from hinge import hinge
from logistic import logistic
from example_tests import example_tests
from vis_spam import vis_spam
# load the data:
data = io.loadmat('data/data_train.mat')
X = data['X']
Y = data['Y']

# split the data:
xTr,xTv,yTr,yTv = valsplit(X,Y)



small_step = 1e-5
feature_vector = np.zeros((xTr.shape[0],1))
lambdaa = 10

ridge_error = checkgradHingeAndRidge(ridge, feature_vector, small_step, xTr, yTr, lambdaa)
print("Ridge error is", ridge_error)

hinge_error = checkgradHingeAndRidge(hinge, feature_vector, small_step, xTr, yTr, lambdaa)
print("Hinge error is", hinge_error)

logistic_error = checkgradLogistic(logistic, feature_vector, small_step, xTr, yTr)
print("Logistic error is", logistic_error)


# for i in range(10):
# 	print((i+1)/50)
# w_trained = trainspamfilter(xTr,yTr)
# spamfilter(xTv,yTv,w_trained,.1)


#print (checkgradHingeAndRidge(ridge,np.zeros((xTr.shape[0],1)),0.1,xTr,yTr,0.1))


#print (checkgradHingeAndRidge(hinge,np.zeros((xTr.shape[0],1)),0.1,xTr,yTr,0.1))

#example_tests()


# vis_spam(w_trained)
