"""
Training of a NN model for learning a model of joint transformations.
Different model complexities are run - single hidden layer with varied number of neurons.

Malte Schilling, 11/14/2018
"""
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from keras.optimizers import SGD
from keras.optimizers import Adam
import numpy as np

import pickle

import matplotlib.pyplot as plt

import scipy.io

######################################
# Parameters ######################### 
######################################
batch_size = 10
epochs = 5000
# Number of repetitions for each architecture
run_training = 5
# Different architectures
hidden_size = [0,1,2,4,8,16,32,64,128]
#hidden_size = [16]
# Store the data in list
hist_list = []

######################################
# Load data for training #############
######################################
mat = scipy.io.loadmat('Data/Overlap_JointAngles_L2L3.mat')
# FIRST LEG AS AN EXAMPLE: 'ANGLES_L2L3'
# TRAIN ON LEG ANGLES
# Complete data set (inputs and targets = number of samples, 6 dimensions of angles
all_data_set_load = mat['ANGLES_L2L3']
# Remove data points with no real angular values (=1.0e+32) - these stem from 
# problem in inverse kinematic calculation
rm_row = []
for i in range(0,all_data_set_load.shape[0]):
    if (np.sum(all_data_set_load[i]) > 1.0e+32):
        rm_row.append(i)
all_data_set = np.delete(all_data_set_load, rm_row, axis=0)
print(all_data_set_load.shape, all_data_set.shape)
#all_data_set /= 180

# Randomly draw indices for training and test set:
indices = np.random.permutation(all_data_set.shape[0])
training_idx, test_idx = indices[:int(0.8 * all_data_set.shape[0])], indices[int(0.8 * all_data_set.shape[0]):]
# Construct training and test set
training_data, test_data = all_data_set[training_idx,:], all_data_set[test_idx,:]

# Cutting training and test set into input data X and target values
train_list = np.hsplit(training_data,2)
X_train = train_list[0]
Targets_train = train_list[1]
test_list = np.hsplit(test_data,2)
X_test = test_list[0]
Targets_test = test_list[1]
model = None
#print(X_train.shape, Targets_train.shape, X_test.shape, Targets_test.shape)

######################################
########## TRAIN NETWORKS ############
######################################
######################################
# Vary Size of Hidden Layer ##########
######################################
for hidd_size in hidden_size:
    print(" ######## HIDDEN MODEL ######## ")
    print(" ######## ", hidd_size)
    print(" ######## HIDDEN MODEL ######## ")
    hist_list.append([])
    # Run multiple runs for each architecture, size of hidden units
    for run_it in range(0, run_training):
        print(" ######## Trainings run ######## ")
        print(" ######## ", run_it)
        print(" ######## HIDDEN MODEL  ######## ")
        print(" ######## ", hidd_size)
        model = Sequential()
        # Adding the hidden layer: fully connected and using sigmoid shaped activation
        # For deeper networks you might switch towards other units if you want.
        if (hidd_size > 0):
            model.add(Dense(hidd_size, activation='sigmoid', input_dim=3))# input_shape=(3,)))
            model.add(Dense(3, activation='linear'))
        # When there is no hidden layer, setup simple linear model
        else:
            print("Model does not include hidden layer")
            model.add(Dense(3, activation='linear', input_dim=3))
        model.summary()

        # Use MSE and Adam as an optimizer
        model.compile(loss='mean_squared_error', 
                  optimizer=Adam())

        # Start training
        history = model.fit(X_train, Targets_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(X_test, Targets_test)) 
        hist_list[-1].append(history.history) 

    
# Structure of the Training data - different levels:
# 1) Top level list = for different architectures (size of hidden layer):
#     [0,1,2,4,8,16,32,64,128]
# 2) Next level list: multiple training runs from random initializations, n=10
# 3) Dict: contains 'loss', 'val_loss' as keys
# 4) and as entries on next level the associated time series (2000 learning iterations)
# Loading the training data from the pickle file
with open('Results/trainHistoryDict_5runs_5000ep_L1L2_backw', 'wb') as file_pi:
    pickle.dump(hist_list, file_pi)       

#print("Sample:     ", X_train[0], Targets_train[0])
#Xnew = np.array([X_train[0]])
# make a prediction
#print("Prediction: ", X_train[0], model.predict(Xnew))
