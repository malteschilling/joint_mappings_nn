"""
Training of a NN model for learning a model of joint transformations.
Applied for a model with skip connections.

Malte Schilling, 11/14/2018
"""

import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, Dropout
#from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np

import pickle

import matplotlib.pyplot as plt

import scipy.io

######################################
# Parameters ######################### 
######################################
batch_size = 10
hidden_size = 4
epochs = 5000
run_training = 5
# Store the data in list
hist_list_4 = []
angles_mse_list_4 = []
hist_list = []
angles_mse_list = []

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

######################################
########## TRAIN NETWORKS ############
######################################
######################################

print(" ######## Training from front to back ######## ")
for run_it in range(0, run_training):
    #create an input tensor 
    inputTensor = Input(shape=(3,))

    #pass the input into the first layer
    firstLayerOutput = Dense(hidden_size, activation='sigmoid')(inputTensor)

    #pass the output through a second layer 
    #secondLayerOutput = Dense(n2)(firstLayerOutput)

    #get the first output and join with the second output (the first output is skipping the second layer)
    skipped = Concatenate()([inputTensor,firstLayerOutput])
    finalOutput = Dense(3, activation='linear')(skipped)

    model_backw_funct = Model(inputTensor,finalOutput)
    model_backw_funct.summary()

    # Use MSE and Adam as an optimizer
    model_backw_funct.compile(loss='mean_squared_error', 
          optimizer=Adam())

    # Start training
    history = model_backw_funct.fit(X_train, Targets_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(X_test, Targets_test)) 
    hist_list_4.append(history.history) 
    
    #print( np.mean(np.square((model_backw_funct.predict(X_test) -Targets_test)), axis=0).shape )
    #print( np.mean(np.square(np.linalg.norm((model_backw_funct.predict(X_test) - Targets_test), axis = 1 )) ) )

    angles_mse_list_4.append(np.mean(np.square((model_backw_funct.predict(X_test) - Targets_test)), axis=0))

print(" ######## Training from front to back ######## ")
for run_it in range(0, run_training):
    #create an input tensor 
    inputTensor = Input(shape=(3,))

    #pass the input into the first layer
    firstLayerOutput = Dense(hidden_size*2, activation='sigmoid')(inputTensor)

    #pass the output through a second layer 
    #secondLayerOutput = Dense(n2)(firstLayerOutput)

    #get the first output and join with the second output (the first output is skipping the second layer)
    skipped = Concatenate()([inputTensor,firstLayerOutput])
    finalOutput = Dense(3, activation='linear')(skipped)

    model_backw_funct = Model(inputTensor,finalOutput)
    model_backw_funct.summary()

    # Use MSE and Adam as an optimizer
    model_backw_funct.compile(loss='mean_squared_error', 
          optimizer=Adam())

    # Start training
    history = model_backw_funct.fit(X_train, Targets_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(X_test, Targets_test)) 
    hist_list.append(history.history) 
    
    #print( np.mean(np.square((model_backw_funct.predict(X_test) -Targets_test)), axis=0).shape )
    #print( np.mean(np.square(np.linalg.norm((model_backw_funct.predict(X_test) - Targets_test), axis = 1 )) ) )

    angles_mse_list.append(np.mean(np.square((model_backw_funct.predict(X_test) - Targets_test)), axis=0))

val_loss = []
for run_hist in hist_list_4:
    val_loss.append(run_hist['val_loss'][-1])
print('Skip Connection training, ', str(epochs), ' ep., ', str(hidden_size), ' hidden:')
print(val_loss)
print('Mean: ', np.mean(val_loss), ' - Std: ', np.std(val_loss))
print(angles_mse_list_4)

val_loss = []
for run_hist in hist_list:
    val_loss.append(run_hist['val_loss'][-1])
print('Skip Connection training, ', str(epochs), ' ep., ', str(hidden_size*2), ' hidden:')
print(val_loss)
print('Mean: ', np.mean(val_loss), ' - Std: ', np.std(val_loss))
print(angles_mse_list)

#file_name = 'transf_model_backw_' + str(hidden_size) + '.h5'
#model_backw.save(file_name)  # creates a HDF5 file 'my_model.h5'
#print("Sample:     ", X_train[0], Targets_train[0])
#Xnew = np.array([X_train[0]])
# make a prediction
#print("Prediction: ", Targets_train[0], model_backw.predict(Xnew))

# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')

# print(" ######## Training from back to front ######## ")
# model_forw = Sequential()
# # Adding the hidden layer: fully connected and using sigmoid shaped activation
# # For deeper networks you might switch towards other units if you want.
# model_forw.add(Dense(hidden_size, activation='sigmoid', input_dim=3))# input_shape=(3,)))
# model_forw.add(Dense(3, activation='linear'))
# model_forw.summary()
# 
# # Use MSE and Adam as an optimizer
# model_forw.compile(loss='mean_squared_error', 
#           optimizer=Adam())
# 
# # Start training
# history = model_forw.fit(Targets_train, X_train, 
#                  batch_size=batch_size,
#                  epochs=epochs,
#                  verbose=1,
#                  validation_data=(Targets_test, X_test)) 
# hist_list.append(history.history) 
# 
# file_name = 'transf_model_forw_' + str(hidden_size) + '.h5'
# model_forw.save(file_name)

# Structure of the Training data - list of
# Dict: contains 'loss', 'val_loss' as keys
# Loading the training data from the pickle file
#with open('trainHistory_2ways', 'wb') as file_pi:
 #   pickle.dump(hist_list, file_pi)       

