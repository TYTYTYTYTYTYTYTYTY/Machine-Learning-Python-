from sklearn.linear_model import Lasso

def lassoReg(x,y,x1):
	clf = Lasso(alpha=0.01,max_iter =100000)
	clf.fit(x,y)

	predict = clf.predict(x1)

	return(predict)