from sklearn.svm import SVC

def svclass(xtr, ytr, xte):
	clf = SVC(C=6.0, cache_size=200, class_weight=None, coef0=0.45,
  decision_function_shape='ovr', degree=11.0, gamma=0.6, kernel='rbf',
 max_iter=-1, probability=True, random_state=42, shrinking=True,
  tol=0.001, verbose=False)
	clf.fit(xtr,ytr)

	predict = clf.predict_proba(xte)

	return(predict[:,1])



	# C=7.8, cache_size=200, class_weight=None, coef0=0.9,
 #  decision_function_shape='ovr', degree=8, gamma=0.2, kernel='poly',
 #  max_iter=-1, probability=False, random_state=42, shrinking=True,
 #  tol=0.001, verbose=False


 # C=5.4, cache_size=200, class_weight=None, coef0=0.3,
 #  decision_function_shape='ovr', degree=10, gamma=0.15, kernel='sigmoid',
 #  max_iter=-1, probability=False, random_state=42, shrinking=True,
 #  tol=0.001, verbose=False


 # C=7.8, cache_size=200, class_weight=None, coef0=1.05,
 #  decision_function_shape='ovr', degree=10, gamma=0.5, kernel='rbf',
 #  max_iter=-1, probability=False, random_state=42, shrinking=True,
 #  tol=0.001, verbose=False      0.9980425055928411


 # C=3.8, cache_size=200, class_weight=None, coef0=1.9,
 #  decision_function_shape='ovr', degree=4.0, gamma=1.75, kernel='poly',
 #  max_iter=-1, probability=False, random_state=42, shrinking=True,
 #  tol=0.001, verbose=False 不拆 0.25

 # C=6.0, cache_size=200, class_weight=None, coef0=0.45,
 #  decision_function_shape='ovr', degree=11.0, gamma=0.6, kernel='rbf',
 #  max_iter=-1, probability=False, random_state=42, shrinking=True,
 #  tol=0.001, verbose=False 不拆 0.25



 # C=6.0, cache_size=200, class_weight=None, coef0=0.45,
 #  decision_function_shape='ovr', degree=11.0, gamma=0.6, kernel='rbf',
 #  max_iter=-1, probability=False, random_state=42, shrinking=True,
 #  tol=0.001, verbose=False 不拆 0.25 


# C=7.8, cache_size=200, class_weight=None, coef0=1.8,
#   decision_function_shape='ovr', degree=6, gamma=1.75, kernel='poly',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False 拆 0.1  0.9944095038434662

# C=7.8, cache_size=200, class_weight=None, coef0=1.05,
#   decision_function_shape='ovr', degree=10, gamma=0.5, kernel='rbf',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False 拆 0.1 0.994


# C=7.8, cache_size=200, class_weight=None, coef0=1.8,
#   decision_function_shape='ovr', degree=6, gamma=1.75, kernel='poly',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False 拆 0.1  0.9958071278825996


# C=6.0, cache_size=200, class_weight=None, coef0=0.4,
#   decision_function_shape='ovr', degree=6, gamma=1.1, kernel='rbf',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False 拆 0.1



# clf = SVC(C=1.4, cache_size=200, class_weight=None, coef0=1.65,
#   decision_function_shape='ovr', degree=12.0, gamma=0.35, kernel='poly',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False) 
# echo



# C=6.0, cache_size=200, class_weight=None, coef0=0.45,
#   decision_function_shape='ovr', degree=11.0, gamma=0.6, kernel='rbf',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False 加了0 0.9963963963963963


# C=3.8, cache_size=200, class_weight=None, coef0=1.9,
#   decision_function_shape='ovr', degree=4.0, gamma=1.75, kernel='poly',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False



# C=7.8, cache_size=200, class_weight=None, coef0=1.05,
#   decision_function_shape='ovr', degree=10, gamma=0.5, kernel='rbf',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False
