#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY
# from naivebayes import naivebayes

def naivebayesCL(x, y):
# =============================================================================
#function [w,b]=naivebayesCL(x,y);
#
#Implementation of a Naive Bayes classifier
#Input:
#x : n input vectors of d dimensions (dxn)
#y : n labels (-1 or +1)
#
#Output:
#w : weight vector
#b : bias (scalar)
# =============================================================================


	
	# Convertng input matrix x and x1 into NumPy matrix
	# input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
	X = np.matrix(x)
	
	# Pre-configuring the size of matrix X
	d,n = X.shape
	
# =============================================================================
# fill in code here
	pos,neg = naivebayesPY(x,y)
	pospost,negpost = naivebayesPXY(x,y)

	pospost = np.array(pospost).reshape((d,1))
	negpost = np.array(negpost).reshape((d,1))

	b = np.log(pos) - np.log(neg)
	w = np.log(pospost) - np.log(negpost)
 
	
	return w,b
# =============================================================================
