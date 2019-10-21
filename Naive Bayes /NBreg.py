from sklearn.naive_bayes import GaussianNB

def NBreg(X_train,Y_train,X_test):

	regressor_GNB = GaussianNB(var_smoothing=1e-3)
	# regressor_GNB = GaussianNB()
	regressor_GNB.fit(X_train,Y_train)


# =============================================================================
# predicting log test vals

	y_obs_GNB = regressor_GNB.predict(X_test)

	return y_obs_GNB



	#0.695e-3