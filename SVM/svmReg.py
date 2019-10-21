from sklearn.svm import SVR
import numpy as np

def svmReg(x,y,x1):
	reg = SVR(gamma='scale', C=1.0, epsilon=0.2 , kernel= 'rbf')
	reg.fit(x, y) 
	predict = reg.predict(x1) 

	return predict



