import numpy as np
import scipy.io as sio
from preprocess import preprocess


bostonData = sio.loadmat('./boston.mat')

xTr = bostonData['xTr']
xTe = bostonData['xTe']

[xTr,xTe, u, m] = preprocess(xTr,xTe)

print(xTr)
yTr = np.array(bostonData['yTr'])
yTe = np.array(bostonData['yTe'])


import matplotlib.pyplot as plt

itr = np.argsort(yTr).flatten();
ite = np.argsort(yTe).flatten();
xTr = xTr[:, itr]
xTe = xTe[:, ite]
yTr = yTr[:, itr]
yTe = yTe[:, ite]

xtr = np.arange(0,np.shape(yTr)[1])
xte = np.arange(0,np.shape(yTe)[1])


plt.plot(xtr, yTr.flatten(), 'r', linewidth=5)
linePredTr = plt.plot(xtr, np.ones(len(xtr)), 'k.')
plt.show()