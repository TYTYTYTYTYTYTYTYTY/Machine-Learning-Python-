# project main 
import numpy as np
import pandas as pd 
# import tensorflow as tf 
import sklearn as sk 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#from scipy import stats
#from scipy.stats import norm, skew 
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Matlab-style plotting
#import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
color = sns.color_palette()
sns.set_style('darkgrid')

#from svclass import svclass

# import keras 
# import torch  


dataset_train = pd.read_csv('train.csv')
print(dataset_train.head(3))
X = dataset_train.iloc[:, 1:12]
n = X.shape[0]
# print(X.head(3))
Y = dataset_train.iloc[:, 12]
# test train split
# # =============================================================================
dataset_test = pd.read_csv('test.csv')
Xtest = dataset_test.iloc[:, 1:12]

X_all = pd.concat([X,Xtest], axis=0)

X_all['Soil_Type'] = X_all['Soil_Type'].astype('category') 
X_all = pd.get_dummies(X_all)



scaler = StandardScaler().fit(np.array(X.iloc[:,[0,1,2,3,4,5,6,7,8,9]]))
X_std = scaler.transform(np.array(X.iloc[:,[0,1,2,3,4,5,6,7,8,9]]))
X_std = pd.DataFrame(X_std)

X_std_full = pd.concat([X_std,X_all.iloc[:n,10::]],axis = 1)
print(list(X_std_full))



X_train, X_test, y_train, y_test = train_test_split(X_std_full, Y, test_size=0.1, random_state=42)












# cross validation
#========================================================================

# from sklearn.model_selection import GridSearchCV # Search over specified parameter values for an estimator.
# from sklearn.model_selection import RandomizedSearchCV # Search over specified parameter values for an estimator.
# from sklearn.model_selection import ShuffleSplit # Random permutation cross-validator
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from AUC import AUC



# classifier = SVC(kernel = 'linear', random_state=42)
# cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
# parameters = {'gamma':(np.arange(1,40)/20).tolist(),
# 			  'degree':(np.arange(1,40)/2).tolist(),
# 			  'C': (np.arange(1,40)/5).tolist(),
# 			  'coef0' :(np.arange(1,40)/20).tolist()}
# scorer = make_scorer(AUC,greater_is_better =True)
# n_iter_search = 10
# grid_obj = RandomizedSearchCV(classifier, 
#                               parameters, 
#                               n_iter = n_iter_search, 
#                               scoring = scorer, 
#                               cv = cv_sets,
#                               random_state= 99)
# grid_fit = grid_obj.fit(X_train, y_train)
# rf_opt = grid_fit.best_estimator_

# print(grid_fit.best_estimator_)




#========================================================================

dataset_test = pd.read_csv('test.csv')
print(dataset_train.head(3))
Xtest = dataset_test.iloc[:, 1:12]
# print(Xtest.head(3))
Xtest['Soil_Type'] = Xtest['Soil_Type'].astype('category') 

scaler = StandardScaler().fit(np.array(X.iloc[:,[0,1,2,3,4,5,6,7,8,9]]))
Xtest_std = scaler.transform(np.array(Xtest.iloc[:,[0,1,2,3,4,5,6,7,8,9]]))
Xtest_std = pd.DataFrame(Xtest_std)
Xtest_std_full = pd.concat([Xtest_std,X_all.iloc[n::,10::]],axis = 1)



ids = dataset_test['ID']
predictions = sklearn_gpc(X_std_full,Y,Xtest_std_full)

output = pd.DataFrame({ 'ID' : ids, 'From_Cache_la_Poudre': predictions })
output.to_csv('svm_submission.csv', index = False)

