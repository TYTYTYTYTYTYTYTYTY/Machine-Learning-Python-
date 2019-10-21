import tensorflow as tf 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Mse import RMse

dataset_train = pd.read_csv('train.csv')
X = dataset_train.iloc[:, 1:11]
print(X.head(3))
Y = dataset_train.iloc[:, 11]
# test train split
# =============================================================================
X['climate_zone'] = X['Soil_Type'].astype(str).str[0] 
X['geologic'] = X['Soil_Type'].astype(str).str[1] 
X['climate_zone'] = X['climate_zone'].astype('category')
X['geologic'] = X['geologic'].astype('category')
X = X.drop(columns =['Soil_Type'])
# X = X.drop(columns =['Aspect'])
# X = X.drop(columns =['Hillshade_9am','Hillshade_3pm'])
# X['dist_hyd'] = np.sqrt(np.power(X.Horizontal_Distance_To_Hydrology,2)+np.power( X.Vertical_Distance_To_Hydrology,2))
# X = X.drop(columns = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'])
# print(X.head(3))

X = pd.get_dummies(X)


# print(X.head(3))

# print(Y.head(3))

X = X.values
Y = Y.values

Y = np.log(Y)




X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25, random_state =0)



# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 16 # data has 10 imput 



Y_train = np.array(Y_train).reshape((5578,1))

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, 1])







    # Hidden fully connected layer with 256 neurons
layer_1 = tf.layers.dense(X,256)
    # Hidden fully connected layer with 256 neurons
layer_2 = tf.layers.dense(layer_1,256, tf.nn.relu)
    # Output fully connected layer with a neuron for each class
output = tf.layers.dense(layer_2,1)



loss = tf.losses.mean_squared_error(Y,output)   # compute cost
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)


sess = tf.Session()                                 # control training and othersb
sess.run(tf.global_variables_initializer())         # initialize var in graph

 # something about plotting
l = 1
step = 1
while l > 0.29:
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], {X: X_train, Y: Y_train})
    step = step+1
    if step % 100 == 0:
    	print(l)
        # plot and show learing process



dataset_train = pd.read_csv('test.csv')
Xtest = dataset_train.iloc[:, 1:11]
# print(X.head(3))
# Y = dataset_train.iloc[:, 11]
# test train split
# =============================================================================
Xtest['climate_zone'] = Xtest['Soil_Type'].astype(str).str[0] 
Xtest['geologic'] = Xtest['Soil_Type'].astype(str).str[1] 
Xtest['climate_zone'] = Xtest['climate_zone'].astype('category')
Xtest['geologic'] = Xtest['geologic'].astype('category')
Xtest = Xtest.drop(columns =['Soil_Type'])


prediction= sess.run(output,{X:X_test})

print(print(RMse(np.exp(prediction),np.exp(Y_test.reshape(-1,1)))))

print(prediction.shape)
print(Y_test.reshape(-1,1).shape)



