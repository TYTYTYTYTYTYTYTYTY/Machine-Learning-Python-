from __future__ import division
import pyGPs
import numpy as np 


def GaussianProcess_C(xtr, ytr, xte):
	models = pyGPs.GPC()
	models.getPosterior(xtr, ytr)
	models.optimize(xtr, ytr)
	predict = models.predict(xte)

	return(Predict) 






