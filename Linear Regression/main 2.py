# project main 
import numpy as np
import pandas as pd 
# import tensorflow as tf 
import sklearn as sk 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from ridgeReg import ridgeReg
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
X = dataset_train.iloc[:, 1:11]
print(X.shape)
print(X.head(3))
Y = dataset_train.iloc[:, 11]

# test train split
# # =============================================================================
X['climate_zone'] = X['Soil_Type'].astype(str).str[0] 
X['geologic'] = X['Soil_Type'].astype(str).str[1] 
X['soil'] = X['Soil_Type'] % 100
# X['soil'] = X['soil'].astype('category')
X['climate_zone'] = X['climate_zone'].astype('category')
X['geologic'] = X['geologic'].astype('category')
X = X.drop(columns =['Soil_Type'])

# X = X.drop(columns =['Aspect'])
# X = X.drop(columns =['Hillshade_3pm'])

print(X.head(3))

# transformer = Normalizer().fit(X.iloc[:,[0,1,2,3,4,5,6,7,10]])
# X_norm = transformer.transform(X.iloc[:,0,1,2,3,4,5,6,7,10])
# X_norm = pd.DataFrame(X_norm)
# X_norm_full = pd.concat([X_norm,X.iloc[:,[8,9]]],axis =1)
# # X = X_norm_full.values

# scaler = StandardScaler().fit(np.array(X.iloc[:,[0,1,2,3,4,5,6,7,8,11]]))
# # print(np.array(X.iloc[:,1:9]).shape)
# X_std = scaler.transform(np.array(X.iloc[:,[0,1,2,3,4,5,6,7,8,11]]))
# X_std = pd.DataFrame(X_std)
# X_std_full = pd.concat([X_std,X.iloc[:,[9,10]]],axis = 1)
# X_std_full = pd.get_dummies(X_std_full)

# # poly = PolynomialFeatures(interaction_only = True, include_bias=False)
# # X_std_full_poly = poly.fit_transform(X_std_full)
# # X_std_full_poly = pd.DataFrame(X_std_full_poly)

# X_std_kernel = RBF(X_std_full).diag

# X_std_kernel = pd.DataFrame(X_std_kernel)

# print(X_std_full_poly.head(3))
# X = X_std_full.values

# X = X.drop(columns =['Hillshade_9am','Hillshade_3pm'])
# X['dist_hyd'] = np.sqrt(np.power(X.Horizontal_Distance_To_Hydrology,2)+np.power( X.Vertical_Distance_To_Hydrology,2))
# X = X.drop(columns = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'])
# print(X.head(3))

# X = pd.get_dummies(X)
# X_norm_full = pd.get_dummies(X_norm_full)
# X_std_full = pd.get_dummies(X_std_full)
# pca = PCA(n_components =5)
# X = pca.fit_transform(X)

# print(X.head(3))

# print(Y.head(3))


# Y = Y.values

# Y = np.log(Y)




# X_train, X_test, Y_train, Y_test = train_test_split(X_std_full_poly,Y,test_size = 0.25, random_state =0)


# from GMM_GMR import GMM_GMR

# gmm = GMM_GMR(4)


#print(trainX.shape)

#print(logReg(trainX,trainY,validateX))
#print(RMse(np.exp(NBreg(X_train,Y_train, X_test)),np.exp(Y_test)))
# print(RMse(np.exp(svmReg(X_train,Y_train,X_test)),Y_test))
# print(RMse(np.exp(lassoReg(X_train,Y_train,X_test)),np.exp(Y_test)))
# print(RMse(np.exp(RandForest(X_train,Y_train,X_test)),np.exp(Y_test)))
# print(RMse(np.exp(ridgeReg(X_train,Y_train,X_test)),np.exp(Y_test)))

# print(RMse(RandForest(X_train,Y_train, X_test),Y_test))
# print(RMse(NBreg(X_train,Y_train, X_test),Y_test))
# print(RMse(lassoReg(X_train,Y_train, X_test),Y_test))
# print(RMse(svmReg(X_train,Y_train, X_test),Y_test))
# print(RMse(ridgeReg(X_train,Y_train, X_test),Y_test))


# for i in range(100):
# 	n = (i+1)/100
# 	print(n)
# # 	# print(Mse(RandForest(X_train,Y_train, X_test,(i+1)*100),Y_test))



# # # for i in range(-40,40):
# # # 	print(0.695+i/200)



# 	mes1 =[]
# 	kf= KFold(n_splits =10, random_state =1 , shuffle= True)

# 	for train_index, test_index in kf.split(X_train):
# 		mes1.append(RMse(ridgeReg(X_train.iloc[train_index], Y_train.iloc[train_index],X_train.iloc[test_index],n),Y_train.iloc[test_index]))

# 	print(np.mean(np.array(mes1)))
# 	print("================")

#========================================================================

# from sklearn.model_selection import GridSearchCV # Search over specified parameter values for an estimator.
# from sklearn.model_selection import RandomizedSearchCV # Search over specified parameter values for an estimator.
# from sklearn.model_selection import ShuffleSplit # Random permutation cross-validator

# rf_regressor = RandomForestRegressor(random_state=42, oob_score = True)
# cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
# parameters = {'max_features':[0.33 ,0.5 ,0.66 ,0.8, None],
# 			  'n_estimators':[200, 400, 600,800,1000], 
#               'min_samples_leaf':[1, 2, 3], 
#               'max_depth':[10,15,20,25,30]}
# scorer = make_scorer(RMse,greater_is_better =False)
# n_iter_search = 10
# grid_obj = RandomizedSearchCV(rf_regressor, 
#                               parameters, 
#                               n_iter = n_iter_search, 
#                               scoring = scorer, 
#                               cv = cv_sets,
#                               random_state= 99)
# grid_fit = grid_obj.fit(X_train, Y_train)
# rf_opt = grid_fit.best_estimator_

# print(grid_fit.best_estimator_)




#========================================================================
# # # # yTe2 = RandForest(xTr,yTr,xTe)
# dataset_test = pd.read_csv('test.csv')
# Xtest = dataset_test.iloc[:, 1:11]

# Xtest['climate_zone'] = Xtest['Soil_Type'].astype(str).str[0] 
# Xtest['geologic'] = Xtest['Soil_Type'].astype(str).str[1] 
# Xtest['soil'] = Xtest['Soil_Type'] % 100
# # Xtest['soil'] = X['soil'].astype('category')
# Xtest['climate_zone'] = Xtest['climate_zone'].astype('category')
# Xtest['geologic'] = Xtest['geologic'].astype('category')
# Xtest = Xtest.drop(columns =['Soil_Type'])
# # Xtest = Xtest.drop(columns =['Hillshade_3pm'])


# # # Xtest = Xtest.drop(columns =['Aspect'])

# # print(X.head(3))

# # transformer = Normalizer().fit(X.iloc[:,[0,1,2,3,4,5,6,7,10]])
# # X_norm = transformer.transform(X.iloc[:,0,1,2,3,4,5,6,7,10])
# # X_norm = pd.DataFrame(X_norm)
# # X_norm_full = pd.concat([X_norm,X.iloc[:,[8,9]]],axis =1)
# # X = X_norm_full.values

# # scaler = StandardScaler().fit(np.array(X.iloc[:,[0,1,2,3,4,5,6,7]]))
# # print(np.array(X.iloc[:,1:9]).shape)
# Xtest_std = scaler.transform(np.array(Xtest.iloc[:,[0,1,2,3,4,5,6,7,8,11]]))
# Xtest_std = pd.DataFrame(Xtest_std)
# Xtest_std_full = pd.concat([Xtest_std,Xtest.iloc[:,[9,10]]],axis = 1)
# Xtest_std_full = pd.get_dummies(Xtest_std_full)

# # Xtest_std_full_poly = poly.fit_transform(Xtest_std_full)
# # Xtest_std_full_poly = pd.DataFrame(Xtest_std_full_poly)
# ids = dataset_test['ID']
# predictions = RandForest(X_std_full,Y,Xtest_std_full)
# # print(Mse(NBreg(X_train,Y_train, X_test),Y_test))

# # # print(list(X_train))

# output = pd.DataFrame({ 'ID' : ids, 'Horizontal_Distance_To_Fire_Points': predictions })
# output.to_csv('submission.csv', index = False)

# # # # # print(yTe2)

# import matplotlib.pyplot as plt  # Matlab-style plotting
# import seaborn as sns
# color = sns.color_palette()
# sns.set_style('darkgrid')



# sns.distplot(Y_train , fit=norm);

# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(Y_train)
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('Horizontal_Distance_To_Fire_Points')

# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(Y_train, plot=plt)
# plt.show()


# preds = np.zeros((5,Xtest.shape[0]))
# preds[0,:] = linear(X_train,Y_train,Xtest)
# preds[1,:] = NBreg(X_train,Y_train,Xtest)
# preds[2,:] = logReg(X_train,Y_train,Xtest)
# preds[3,:] = svmReg(X_train,Y_train,Xtest)
# preds[4,:] = lassoReg(X_train,Y_train,Xtest)

# # print(Mse(np.mean(preds, axis=0), Y_test))

# ids = dataset_test['ID']
# predictions = np.mean(preds, axis=0)


# print(list(X_train))

# output = pd.DataFrame({ 'ID' : ids, 'Horizontal_Distance_To_Fire_Points': predictions })
# output.to_csv('submission.csv', index = False)






