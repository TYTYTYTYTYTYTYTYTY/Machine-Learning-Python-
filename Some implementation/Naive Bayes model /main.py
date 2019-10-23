import sys
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY
from naivebayes import naivebayes
import numpy as np 
from naivebayesCL import naivebayesCL
from genTrainFeatures import genTrainFeatures
from whoareyou import whoareyou
from classifyLinear import classifyLinear
y=np.matrix([-1, 1])
x=np.matrix([[0, 1], [1, 0]])

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
Ynew = np.hstack((Y, Y0))

d,n = Xnew.shape

# print(Ynew.shape)

# print(n)

# print(2/4)

# print(np.where(Ynew ==1,1,0).sum()/n)



# print(naivebayesPXY(x,y))
# Ynew = np.squeeze(Ynew)
# print(Ynew)
# print(Xnew[1,:])

# print(np.diagonal(Xnew[1,:].T*Ynew))

# print((np.where(Ynew==1,1,0).flatten() * Xnew.T).sum())
# print(naivebayes(x,y,Xnew[:,1]))

# xTr,yTr = genTrainFeatures() 
# w,b = naivebayesCL(xTr,yTr) 
# # print(w.shape)
# # print(b.shape)
# whoareyou(w,b)


#print(np.matrix.tolist(xTr.T))
# print(xTr[:,1].shape)
# print(naivebayes(xTr,yTr,np.matrix.tolist(xTr.T)[1]))

# print(np.array(np.matrix.tolist(xTr.T)[1]).reshape((128,1)))

# pospost,negpost = naivebayesPXY(xTr,yTr)

# pospost = np.array(pospost).reshape((128,1))
# negpost = np.array(negpost).reshape((128,1))
# X1 = np.array(np.matrix.tolist(xTr.T)[1]).reshape((128,1))

# print(pospost.shape)
# print(np.where(X1!=0))
# print(pospost[np.where(X1!=0)])

[x,y]=genTrainFeatures()
[w,b]=naivebayesCL(x,y)
preds=classifyLinear(x,w,b)


# y1 = list(map(lambda x1: naivebayes(x,y,x1),np.matrix.tolist(np.matrix(x.T))))
# y1 = np.array(y1).reshape((1,1200))
# y1 = np.where(y1 > 0 ,1,-1)
trainingerror = np.sum(preds!=y)/(y.shape[1])

print(trainingerror)







