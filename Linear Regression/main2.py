# project main 
import numpy as np
import pandas as pd 
# import tensorflow as tf 
import sklearn as sk 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from scipy import stats
from scipy.stats import norm, skew 
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
color = sns.color_palette()
sns.set_style('darkgrid')

# import keras 
# import torch  


dataset_train = pd.read_csv('train.csv')
print(dataset_train.head(3))
X = dataset_train.iloc[:, 1:12]
# print(X.head(3))
Y = dataset_train.iloc[:, 12]
# test train split
# # =============================================================================
X['climate_zone'] = X['Soil_Type'].astype(str).str[0]
X['climate_zone'] = X['climate_zone'].astype(float)
X['geologic'] = X['Soil_Type'].astype(str).str[1]
X['geologic'] = X['geologic'].astype(float)
X['soil'] = X['Soil_Type'] % 100
X['soil'] = X['soil'].astype(float)
# X['soil'] = X['soil'].astype('category')
# print(X['soil'])
# X['climate_zone'] = X['climate_zone'].astype('category')
# X['geologic'] = X['geologic'].astype('category')
X = X.drop(columns =['Soil_Type'])

# X = X.drop(columns =['Aspect'])
# X = X.drop(columns =['Hillshade_3pm'])

print(X.head(3))

# transformer = Normalizer().fit(X.iloc[:,[0,1,2,3,4,5,6,7,10]])
# X_norm = transformer.transform(X.iloc[:,0,1,2,3,4,5,6,7,10])
# X_norm = pd.DataFrame(X_norm)
# X_norm_full = pd.concat([X_norm,X.iloc[:,[8,9]]],axis =1)
# X = X_norm_full.valuesd

scaler = StandardScaler().fit(np.array(X))
# print(np.array(X.iloc[:,1:9]).shape)
X_std = scaler.transform(np.array(X))
X_std_full = pd.DataFrame(X_std)
# X_std_full = pd.concat([X_std,X.iloc[:,[10,11,12]]],axis = 1)
# X_std_full = pd.get_dummies(X_std_full)


print(X_std_full.head(3))


X_train, X_test, y_train, y_test = train_test_split(X_std_full, Y, test_size=0.1, random_state=42)



# # poly = PolynomialFeatures(interaction_only = True, include_bias=False)
# # X_std_full_poly = poly.fit_transform(X_std_full)
# X_std_full_poly = pd.DataFrame(X_std_full_poly)

# X_std_kernel = RBF(X_std_full).diag

# X_std_kernel = pd.DataFrame(X_std_kernel)

# print(X_std_full_poly.head(3))












#========================================================================

# from sklearn.model_selection import GridSearchCV # Search over specified parameter values for an estimator.
# from sklearn.model_selection import RandomizedSearchCV # Search over specified parameter values for an estimator.
# from sklearn.model_selection import ShuffleSplit # Random permutation cross-validator
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from AUC import AUC



# classifier = SVC(kernel = 'rbf', random_state=42, cache_size = 1000)
# cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
# parameters = {'gamma':(np.arange(1,40)/20).tolist(),
# 			  'degree':np.arange(1,20).tolist(),
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



from sklearn_gpc import sklearn_gpc

dataset_test = pd.read_csv('test.csv')
print(dataset_train.head(3))
Xtest = dataset_test.iloc[:, 1:12]
# print(X.head(3))


# # =============================================================================
Xtest['climate_zone'] = Xtest['Soil_Type'].astype(str).str[0] 
Xtest['geologic'] = Xtest['Soil_Type'].astype(str).str[1] 
Xtest['soil'] = Xtest['Soil_Type'] % 100
# X['soil'] = X['soil'].astype('category')
Xtest['climate_zone'] = X['climate_zone'].astype('float')
Xtest['geologic'] = Xtest['geologic'].astype('float')
Xtest = Xtest.drop(columns =['Soil_Type'])



# print(np.array(X.iloc[:,1:9]).shape)
Xtest_std = scaler.transform(np.array(Xtest))
Xtest_std_full = pd.DataFrame(Xtest_std)

# print(Xtest_std_full.head(3))

ids = dataset_test['ID']
print(sklearn_gpc(X_std_full.head(100),Y.head(100),Xtest_std_full.head(100)))

predictions = sklearn_gpc(X_std_full,Y,Xtest_std_full)



output = pd.DataFrame({ 'ID' : ids, 'From_Cache_la_Poudre': predictions })
output.to_csv('gp_xiagao_submission.csv', index = False)

