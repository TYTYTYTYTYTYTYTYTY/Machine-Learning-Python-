import numpy as np 
from sklearn import linear_model


def logReg(x,y,x1):
	reg = linear_model.LogisticRegression(random_state=0)
	reg.fit(x,y)

	predict = reg.predict(x1)

	return(predict)
