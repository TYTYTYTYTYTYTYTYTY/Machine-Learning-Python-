"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    bias = 0
    #print(alphas)
    s=min(min(np.where(((alphas[:,0]-0)>0.00001) & ((alphas[:,0]-C)<-0.00001))))
    #print(s)

    #print(yTr*alphas)
    bias = yTr[s,:] -  np.matmul(K[s,:],yTr*alphas)


    # print(bias)
    
    
    

    
    
    
    return bias 
    
