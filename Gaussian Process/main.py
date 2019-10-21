# project main 
import numpy as np
import pandas as pd 
import tensorflow as tf 
import sklearn as sk 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from scipy import stats
from scipy.stats import norm, skew 
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
color = sns.color_palette()
sns.set_style('darkgrid')



# import keras 
# import torch  


dataset_train = pd.read_csv('train.csv', )
print(dataset_train.head(3))
X = dataset_train.iloc[:, 1:13]
n = X.shape[0]
# print(X.head(3))
Y = dataset_train.iloc[:, 13]
# test train split


X['sin'] = np.sin( X['Aspect'])
X['cos'] = np.cos( X['Aspect'])
X = X.drop(columns =['Aspect'])


X['climate_zone'] = X['Soil_Type'].astype(str).str[0] 
X['geologic'] = X['Soil_Type'].astype(str).str[1] 
X['soil'] = X['Soil_Type'] % 100
X['soil'] = X['soil'].astype('category')
X['climate_zone'] = X['climate_zone'].astype('category')
X['geologic'] = X['geologic'].astype('category')
X = X.drop(columns =['Soil_Type'])


X['Wilderness_Area1'] = X['Wilderness_Area'].astype('category')
X = X.drop(columns =['Wilderness_Area'])

X.iloc[:,:11] = X.iloc[:,:11].astype('float32')




X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


print(X_train.iloc[:,:11].head(3))

minmax = MinMaxScaler().fit(X_train.iloc[:,:11])
X_train_mm = minmax.transform(X_train.iloc[:,:11]) 
X_test_mm = minmax.transform(X_test.iloc[:,:11])


X_train.iloc[:,:11] = X_train_mm
X_test.iloc[:,:11] = X_test_mm

X_train_mm_oh = pd.get_dummies(X_train)
# print(X_train_mm_oh.head(3))
X_test_mm_oh  = pd.get_dummies(X_test)
# print(X_test_mm_oh.head(3))


y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test)

print(np.argmax(y_train, axis=1))
print(list(y_test))
# num_labels = 


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, axis=1))
            / predictions.shape[0])







#=============================================================================================
# Parameters
# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# display_step = 1

# # Network Parameters
# n_hidden_1 = 384 # 1st layer number of neurons
# n_hidden_2 = 256 # 2nd layer number of neurons
# n_hidden_3 = 256
# n_hidden_4 = 128
# n_input = 51 # data has 10 imput 



# X = tf.placeholder("float", [None, n_input])
# Y = tf.placeholder("float", [None, 7])



# layer_1 = tf.layers.dense(X,n_hidden_1 , tf.nn.relu)
    
# layer_2 = tf.layers.dense(layer_1, n_hidden_2, tf.nn.relu)
   
# layer_3 = tf.layers.dense(layer_2, n_hidden_3, tf.nn.relu)

# layer_4 = tf.layers.dense(layer_3, n_hidden_4, tf.nn.relu)

# logits = tf.layers.dense(layer_4,7)



# loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=Y))  # compute cost

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# train_op = optimizer.minimize(loss)

# predictions = tf.nn.softmax(logits)





# sess = tf.Session()                                 # control training and othersb
# sess.run(tf.global_variables_initializer())         # initialize var in graph


# for step in range(2001):
#         _, l, pred = sess.run([train_op, loss, predictions], {X: X_train_mm_oh, Y: y_train})
#         if step % 50 == 0:
#             print('step:{} loss:{:.6f} accuracy: {:.2f}'.format(
#                     step, l, accuracy(pred, y_train)))



# prediction= sess.run(prediction,{X:X_test_mm_oh})




# num_node1 = X_train_mm_oh.shape[1]
# num_labels = y_train.shape[1]
# learning_rate =0.1



# graph = tf.Graph()

# with graph.as_default():
#     tf_train_dataset = tf.constant(X_train_mm_oh)
#     tf_train_labels = tf.constant(y_train)

#     weights = tf.cast(tf.Variable(tf.truncated_normal([num_node1, num_labels])),tf.float64)
#     biases = tf.cast(tf.Variable(tf.zeros([num_labels])), tf.float64)

#     logits = tf.matmul(tf_train_dataset, weights) + biases
#     loss = tf.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=tf_train_labels))

#     optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#     train_prediction = tf.nn.softmax(logits)




# with tf.Session(graph=graph) as session:
#     tf.initialize_all_variables().run()
#     for step in range(10001):
#         _, l, predictions = session.run([optimizer, loss, train_prediction])
#         if step % 500 == 0:
#             print('step:{} loss:{:.6f} accuracy: {:.2f}'.format(
#                     step, l, accuracy(predictions, y_train)))
















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

# dataset_test = pd.read_csv('test.csv')
# print(dataset_train.head(3))
# Xtest = dataset_test.iloc[:, 1:12]
# # print(Xtest.head(3))
# Xtest['Soil_Type'] = Xtest['Soil_Type'].astype('category') 

# scaler = StandardScaler().fit(np.array(X.iloc[:,[0,1,2,3,4,5,6,7,8,9]]))
# Xtest_std = scaler.transform(np.array(Xtest.iloc[:,[0,1,2,3,4,5,6,7,8,9]]))
# Xtest_std = pd.DataFrame(Xtest_std)
# Xtest_std_full = pd.concat([Xtest_std,X_all.iloc[n::,10::]],axis = 1)



# ids = dataset_test['ID']
# predictions = svclass(X_std_full,Y,Xtest_std_full)

# output = pd.DataFrame({ 'ID' : ids, 'From_Cache_la_Poudre': predictions })
# output.to_csv('svm_submission.csv', index = False)

