import numpy as np 
from sklearn.linear_model import LinearRegression 


def linear(train_x, train_y, predict_x): 
	reg= LinearRegression().fit(train_x, train_y)
	prediction = reg.predict(predict_x)

	return(prediction)
