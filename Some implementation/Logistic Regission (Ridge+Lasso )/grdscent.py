
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
	eps = 2.2204e-14 #minimum step size for gradient descent
	# initialization 
	w_old = w0
	old_grade = func(w_old)[1]
	old_loss = func(w_old)[0]
	n =0 

	while n <= maxiter:

		if np.linalg.norm(old_grade) <= tolerance :
			break

		w = w_old - stepsize*old_grade
		new_loss = func(w)[0]

		if new_loss <= old_loss :
			stepsize = stepsize*1.01
		else :
			if stepsize*0.5 < eps:
				pass
			else : 
				stepsize = stepsize*0.5 

		old_grade = func(w)[1]
		old_loss = new_loss 
		w_old = w 
		n = n+1
	#print(stepsize)
	return w
