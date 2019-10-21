import numpy as np 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets import make_regression 

def RandForest(train_x, train_y, test_x):
	rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=25,
           max_features=0.8, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
           oob_score=True, random_state=42, verbose=0, warm_start=False)
	rf.fit(train_x,train_y)
	prediction = rf.predict(test_x)

	# print(rf.feature_importances_)

	return(prediction)


# 0.66 332.622
# n =400