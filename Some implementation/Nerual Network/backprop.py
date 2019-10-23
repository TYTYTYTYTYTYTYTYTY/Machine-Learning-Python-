# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:07:53 2019

@author: Jerry Xing
"""
import numpy as np
def backprop(W, aas,zzs, yTr,  trans_func_der):
#% function [gradient] = backprop(W, aas, zzs, yTr,  der_trans_func)
#%
#% INPUT:
#% W weights (list of ndarray)
#% aas output of forward pass (list of ndarray)
#% zzs output of forward pass (list of ndarray)
#% yTr 1xn ndarray (each entry is a label)
#% der_trans_func derivative of transition function to apply for inner layers
#%
#% OUTPUTS:
#% 
#% gradient = the gradient at w as a list of ndarries
#%

	n = np.shape(yTr)[1]
	# print(W[1].shape)
	# print(zzs)
	delta = (zzs[0] - yTr).reshape(1,-1)
	# print(delta[:,0].shape)
	# print(delta.shape)
	# print(aas[1].shape)
	
	# compute gradient with back-prop
	# gradient = [None] * len(W)
	# gradient[0] = delta
	# for i in range(len(W)):
	# 	print(aas[i+1].shape)
	# 	print(W[i][:,1::].T.shape)
	# 	print(())
	# 	print(delta.shape)
	# 	print('===========')

	# 	delta= trans_func_der(aas[i+1])* W[i][:,1::].T @ delta
	# 	gradient[i+1]=delta
		# INSERT CODE HERE:



	gradient = []
	for i in range(len(W)):
		gradient.append(W[i]*0)

	



# gradient = []
#  for i in range(len(W)):
#   gradient.append(W[i]*0)

#  #print(gradient[0])
#  gradient[0]=gradient[0]+np.mean(delta*zzs[1],axis=1).reshape((1,zzs[1].shape[0]))
#  print(gradient[0])
#  #print(np.mean(delta*zzs[1],axis=1).reshape((1,zzs[1].shape[0])))

#  for i in range(n):
#   delta_temp = delta[:,i]
  
#   for j in range(1,len(W)):
#    #delta_temp = W[j-1][:,0:(W[j-1].shape[1]-1)] @ (trans_func_der(aas[j][:,i]).reshape(aas[j].shape[0],1)*delta_temp)
#    print(delta_temp.shape)
#    print(((W[j-1][:,0:(W[j-1].shape[1]-1)].T)@ delta_temp).shape)
#    delta_temp = ((W[j-1][:,0:(W[j-1].shape[1]-1)].T) @ delta_temp).reshape((aas[j].shape[0],1))*trans_func_der(aas[j][:,i]).reshape((aas[j].shape[0],1))
   
#    print(delta_temp.shape)
#    #gradient[j][:,0:(gradient[j].shape[1]-1)] +=  1/n*(zzs[j+1][0:zzs[j+1].shape[0]-1,i].reshape(zzs[j+1].shape[0]-1,1) @ delta_temp.T).T
#    gradient[j][:,0:(gradient[j].shape[1]-1)] +=  1/n*(delta_temp @ (zzs[j+1][0:zzs[j+1].shape[0]-1,i].reshape(zzs[j+1].shape[0]-1,1)).T )
#    gradient[j][:,gradient[j].shape[1]-1] += 1/n*np.squeeze( delta_temp) 


#  return gradient
	
	delta_temp = delta
	# print(zzs[1][:-1,:].shape)
	# print(delta_temp.shape)
	gradient[0][:,:-1] += 1/n* (delta_temp @ zzs[1][:-1,:].reshape(zzs[1].shape[0]-1,-1).T)
	gradient[0][:,-1] += 1/n* (delta_temp @ zzs[1][-1,:].reshape(1,-1).T).flatten()

	for i in range(1,len(W)):
		delta_temp  = (W[i-1][:,:-1].T @ delta_temp ) * trans_func_der(aas[i]).reshape(aas[i].shape[0],-1)
		gradient[i][:,:-1] += 1/n* delta_temp @ zzs[i+1][:-1,:].reshape(zzs[i+1].shape[0]-1,-1).T
		gradient[i][:,-1] += 1/n* (delta_temp @ zzs[i+1][-1,:].reshape(1,-1).T ).flatten()

	return gradient 


