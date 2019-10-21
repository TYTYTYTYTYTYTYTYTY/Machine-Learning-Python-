import numpy as np
from sklearn.kernel_ridge import KernelRidge 


def  ridgeReg(x,y,x1):
	clf = KernelRidge(alpha =0.4)
	clf.fit(x,y)

	predict = clf.predict(x1)

	return predict

