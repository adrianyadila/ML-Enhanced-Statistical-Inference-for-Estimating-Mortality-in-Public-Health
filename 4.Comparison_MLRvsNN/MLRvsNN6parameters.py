# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import layers
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.utils import plot_model
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from keras.models import load_model
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-csv", "--datasetcsv", required=True,
	help="path to input dataset")
#ap.add_argument("-model", "--modelNNsixparameters", required=True,
	#help="path to output label binarizer")
ap.add_argument("-hist", "--hNNsixparameters", required=True,
	help="path to output label binarizer")
args = vars(ap.parse_args())

# retrieve:    
f = open(args["hNNsixparameters"], 'rb')
H = pickle.load(f)
f.close()

print("[INFO] loading neural network...")
modelNNsix = load_model(
    r"C:/Users/Latitude 5490/OneDrive\Documentos/NN_sixparameters_save_SGD_momentum.h5", 
    custom_objects={"r2_score": r2_score}
)
#modelNNsix = load_model(r"C:/Users/Latitude 5490/OneDrive/Documentos/NN_sixparameters_save_pat30.h5")
#modelNNsix = load_model("C:\Users\Latitude 5490\OneDrive\Documentos\NN_sixparameters_save_pat30.h5")

df= pd.read_csv(args["datasetcsv"])
print(df.head())

# Counting NaN values in all columns
nan_count = df.isna().sum().sum()

print('number of missing values for all variables:')
print(nan_count)
print(df.shape)

print('percentage of missing values for all variables:')
print(nan_count*100/(df.shape[0]*df.shape[1]))

#excluding variables with huge number of missing values
df2 = df.loc[:, ["var 122", "var 93", "var 48", "var 106", "var 35", "var 34", "var 10"]] 

# Counting NaN values for new dataset
nan_count2 = df2.isna().sum().sum()

print('number of missing values for selected variables:')
print(nan_count2)

print(df2.shape)

print(df2.head())

print('percentage of missing values for selected variables:')
nan_count2*100/(df2.shape[0]*df2.shape[1])

# Finding the mean of the column having NaN
for i in df2.columns:
  mean_value = []
  if df2[i].isna().sum() > 0:
    # Replace NaNs in column S2 with the
    mean_value = df2[i].mean()

    # Replace NaNs in column S2 with the
    # mean of values in the same column
    df2[i].fillna(value=mean_value, inplace=True)

print(df2.head())

#CREATING PREDICTOR AND TARGET SETS

target = df2.iloc[:, 6].values
print('target:')
print(target)
print(target.shape)

predictor = df2.drop('var 10', axis=1)
predictor = predictor.values
print('predictor:')
print(predictor)
print(target.shape)

#NORMALIZATION

predictor_norm = StandardScaler().fit_transform(predictor)
print(predictor_norm)

# Reshape the target array to be 2D
target_2d = target.reshape(-1, 1)

# Now you can apply StandardScaler
target_norm = StandardScaler().fit_transform(target_2d)

# If you want to convert it back to a 1D array:
target_norm_1d = target_norm.flatten()
print(target_norm_1d)

#CREATING TRAINING AND TEST SETS

x_train, x_test, y_train, y_test = train_test_split(predictor_norm, target_norm_1d, test_size = 0.3, random_state = 0)

print('x_train shape:')
print(x_train.shape)

print('x_test shape:')
print(x_test.shape)

print('y_train shape:')
print(y_train.shape)

print('y_test shape:')
print(y_test.shape)

x_test_trp = x_test.transpose()
print(x_test_trp[0])

y_pred = modelNNsix.predict(x_test)

y_pred_train = modelNNsix.predict(x_train)

plot_model(modelNNsix, to_file='model_six_final.png', show_shapes=True, show_layer_names=True)

#MULTIPLE LINEAR REGRESSION MODEL

reg = LinearRegression().fit(x_train, y_train)
print('r2_score:')
print(reg.score(x_train, y_train))

print('Coefficients:')
print(reg.coef_)

print('Intercept:')
print(reg.intercept_)

# Make predictions using the test set
y_pred_train_mlr = reg.predict(x_train)
y_pred_mlr = reg.predict(x_test)

#RESIDUALS for NN and MLR 

y_pred = y_pred.squeeze()
y_pred_train = y_pred_train.squeeze()

#RESIDUALS TEST
residuals_NN = np.subtract(y_test, y_pred) 
residuals_MLR = np.subtract(y_test, y_pred_mlr)  

# Plot outputs
plt.figure()
plt.scatter(y_pred, residuals_NN, s=5, color="green")
plt.axhline(y=0, color='black')
plt.xlabel('predicted value', fontsize=15)
plt.ylabel('residuals', fontsize=15)
plt.show()

# Plot outputs
plt.figure()
plt.scatter(y_pred, residuals_NN, s=5, color="green")
plt.scatter(y_pred_mlr, residuals_MLR, s=5, color="blue")
plt.legend(["neural network" , "multiple linear regression"])
plt.axhline(y=0, color='black')
plt.xlabel('predicted value', fontsize=15)
plt.ylabel('residuals', fontsize=15)
plt.show()

#RESIDUALS TRAIN
residuals_NN_train = y_train - y_pred_train
residuals_MLR_train = y_train - y_pred_train_mlr

# Plot outputs
plt.figure()
plt.scatter(y_pred_train, residuals_NN_train, s=5, color="green")
plt.axhline(y=0, color='black')
plt.xlabel('predicted value', fontsize=15)
plt.ylabel('residuals', fontsize=15)
plt.show()

# Plot outputs
plt.figure()
plt.scatter(y_pred_train, residuals_NN_train, s=5, color="green")
plt.scatter(y_pred_train_mlr, residuals_MLR_train, s=5, color="blue")
plt.legend(["neural network" , "multiple linear regression"])
plt.axhline(y=0, color='black')
plt.xlabel('predicted value', fontsize=15)
plt.ylabel('residuals', fontsize=15)
plt.show()

#Predicted vs Observed

# Determine global axis limits
global_min = min(min(y_pred_train), min(y_train), min(y_pred), min(y_test))
global_max = max(max(y_pred_train), max(y_train), max(y_pred), max(y_test))

# Set the same axis limits for a square figure
axis_limits = [global_min, global_max]

# Plot outputs for training data
plt.figure(figsize=(6, 6))  # Ensure the figure is square
plt.scatter(y_pred_train, y_train, s=5, color="green")
plt.plot(axis_limits, axis_limits, color="black", linewidth=2)  # Diagonal line
plt.xlim(axis_limits)
plt.ylim(axis_limits)
plt.xlabel("predicted", fontsize=15)
plt.ylabel("observed", fontsize=15)
plt.show()

# Plot outputs for training data
plt.figure(figsize=(6, 6))  # Ensure the figure is square
plt.scatter(y_pred_train, y_train, s=5, color="green")
plt.scatter(y_pred_train_mlr, y_train, s=5, color="blue")
plt.legend(["neural network", "multiple linear regression"])
plt.plot(axis_limits, axis_limits, color="black", linewidth=2)  # Diagonal line
plt.xlim(axis_limits)
plt.ylim(axis_limits)
plt.xlabel("predicted", fontsize=15)
plt.ylabel("observed", fontsize=15)
plt.show()

# Plot outputs for test data
plt.figure(figsize=(6, 6))  # Ensure the figure is square
plt.scatter(y_pred, y_test, s=5, color="green")
plt.plot(axis_limits, axis_limits, color="black", linewidth=2)  # Diagonal line
plt.xlim(axis_limits)
plt.ylim(axis_limits)
plt.xlabel("predicted", fontsize=15)
plt.ylabel("observed", fontsize=15)
plt.show()

# Plot outputs for test data
plt.figure(figsize=(6, 6))  # Ensure the figure is square
plt.scatter(y_pred, y_test, s=5, color="green")
plt.scatter(y_pred_mlr, y_test, s=5, color="blue")
plt.legend(["neural network", "multiple linear regression"])
plt.plot(axis_limits, axis_limits, color="black", linewidth=2)  # Diagonal line
plt.xlim(axis_limits)
plt.ylim(axis_limits)
plt.xlabel("predicted", fontsize=15)
plt.ylabel("observed", fontsize=15)
plt.show()

#plot loss functions
plt.plot(H['r2_score'])
plt.plot(H['val_r2_score'])
plt.ylabel('r2_score')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



