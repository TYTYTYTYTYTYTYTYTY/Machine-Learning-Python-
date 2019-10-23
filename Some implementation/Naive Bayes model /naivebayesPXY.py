#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np

def naivebayesPXY(x, y):
# =============================================================================
#    function [posprob,negprob] = naivebayesPXY(x,y);
#
#    Computation of P(X|Y)
#    Input:
#    x : n input vectors of d dimensions (dxn)
#    y : n labels (-1 or +1) (1xn)
#    
#    Output:
#    posprob: probability vector of p(x|y=1) (dx1)
#    negprob: probability vector of p(x|y=-1) (dx1)
# =============================================================================


    
    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    Y = np.matrix(y)
    
    # Pre-configuring the size of matrix X
    d,n = X.shape
    
    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2))
    Y0 = np.matrix('-1, 1')
    
    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise)
        #Xnew = np.concatenate((X, X0), axis=1) #concatenate to column
    Ynew = np.hstack((Y, Y0))
    
    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape
    
# =============================================================================
# fill in code here
    posprob = []
    negprob = []

    for i in range(d):
        posprob.append((np.where(Ynew==1,1,0).flatten() * Xnew[i,:].T).sum()/(np.where(Ynew==1,1,0).flatten() * Xnew.T).sum())
        negprob.append((np.where(Ynew==-1,1,0).flatten() * Xnew[i,:].T).sum()/(np.where(Ynew==-1,1,0).flatten() * Xnew.T).sum() )
    
    posprob = np.array(posprob).reshape((d,1))
    negprob = np.array(negprob).reshape((d,1))
    return posprob,negprob

# =============================================================================
