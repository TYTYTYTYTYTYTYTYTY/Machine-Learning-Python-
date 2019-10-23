"""
INPUT:  
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in cvxopt.solvers.qp

A call of cvxopt.solvers.qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays and keep the return 
statement as is so they are in the right format. See this reference to assign variables:
https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf.
"""
import numpy as np
from cvxopt import matrix

def generateQP(K, yTr, C):
	yTr = yTr.astype(np.double)
	n = yTr.shape[0]
	yTr = np.squeeze(yTr)
	
	p = np.ones((n,1))*-1

	Q = np.matmul(np.matmul(np.diag(yTr), K),np.diag(yTr))

	G_up =np.diag(np.ones(n)* -1)
	G_down = np.diag(np.ones(n))
	G = np.concatenate((G_up, G_down), axis=0)

	h_up = np.zeros((n,1))
	h_down = np.ones((n,1))*C
	h = np.concatenate((h_up, h_down), axis =0)



	A = np.array(yTr).reshape((1,n))
	b = np.zeros(1).reshape((1,1))

			
	return matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)

