"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm
from sklearn.model_selection import KFold

def crossvalidate(xTr, yTr, ktype, Cs, paras):
	bestC, bestP, lowest_error = 0, 0, 0
	errors = np.zeros((len(paras),len(Cs)))
	variance = np.zeros((len(paras),len(Cs)))
	n = yTr.shape[0]
	n_fold =100

	kf= KFold(n_splits =10, random_state = 666, shuffle= True)

	lowest_error=1 
	lowest_var =100

	for i in range(len(Cs)):
		for j in range(len(paras)):
			err =[]
			for train_index, test_index in kf.split(xTr.T):
				svmclassify = trainsvm(xTr[:,train_index], yTr[train_index,:], Cs[i], ktype, paras[j])
				preds = svmclassify(xTr[:,test_index])
				err.append(np.mean(preds != yTr[test_index,:]))
			variance[j,i] = np.var(np.array(err))
			errors[j,i] = np.mean(np.array(err))
			# print(errors[j,i])

			if errors[j,i] < lowest_error: 
				lowest_error = errors[j,i]
				lowest_var = variance[j,i]
				# print(lowest_error)
				bestC = Cs[i]
				bestP = paras[j]
			else :
				if errors[j,i] == lowest_error and variance[j,i] < lowest_var: 
					bestC = Cs[i]
					bestP = paras[j]


	
	return bestC, bestP, lowest_error, errors


	