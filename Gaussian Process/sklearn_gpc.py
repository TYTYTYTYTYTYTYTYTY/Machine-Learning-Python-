from sklearn.gaussian_process import GaussianProcessClassifier 
from sklearn.gaussian_process.kernels import RBF 

def sklearn_gpc(xtr, ytr, xte):
	kernel = 1.0*RBF(1.0)
	gpc = GaussianProcessClassifier(kernel= kernel, random_state =666, n_jobs =-1)

	gpc.fit(xtr, ytr)

	predict = gpc.predict_proba(xte)[:,1]
	# predict = gpc.predict(xte)

	return(predict)



# 1,2 |0.7 0.9841174707821396

# 0.1 ,0.1 | 0.7 0.9841174707821396

# 1,1,|0.7 0.9841174707821396