from sklearn import metrics 
import numpy as np 


def AUC (ypred, ytrue):
	fpr, tpr, thresholds = metrics.roc_curve(ytrue ,ypred, pos_label=1)
	return(metrics.auc(fpr, tpr))