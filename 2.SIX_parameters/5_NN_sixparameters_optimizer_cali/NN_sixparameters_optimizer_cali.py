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
from keras.optimizers import SGD

ap = argparse.ArgumentParser()
ap.add_argument("-csv", "--datasetcsv", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model",  required=True,
	help="path to output model")
ap.add_argument("-hist", "--history", required=True,
	help="path to output model's history")
args = vars(ap.parse_args())

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

#NEURAL NETWORK WITH KERAS

model = Sequential()
model.add(Dense(100, input_dim=6, activation = 'relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation = 'linear'))

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=30, min_lr=1e-6, verbose=1)

# Define the SGD optimizer with momentum
optimizer = SGD(learning_rate=0.01, momentum=0.8)  # Adjust learning_rate and momentum as needed

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[r2_score], run_eagerly=True)

H = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=[reduce_lr])

plot_model(model, to_file='model_six.png', show_shapes=True, show_layer_names=True)

y_pred = model.predict(x_test)

y_pred_train = model.predict(x_train)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the history:
print("[INFO] serializing history...")
f = open(args["history"], 'wb')
pickle.dump(H.history, f)
f.close()

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

residuals_NN = np.subtract(y_test, y_pred) 
#residuals_NN = np.reshape(residuals_NN, -1)
residuals_MLR = np.subtract(y_test, y_pred_mlr)  

print('shapes primeiro')
print(residuals_NN.shape)
print(y_pred.shape)
print(y_test.shape)

print('shapes segundo')
print(residuals_MLR.shape)
print(y_pred.shape)

# Plot outputs
plt.scatter(y_pred, residuals_NN, s=5, color="blue")
plt.scatter(y_pred_mlr, residuals_MLR, s=5, color="orange")
plt.legend(["Neural Network" , "Multiple Linear Regression"])
plt.axhline(y=0, color='b')
plt.xlabel('Predicted', fontsize=15)
plt.ylabel('Residuals', fontsize=15)
plt.title('Predicted vs Residuals - Test (6 var)')
plt.show()

residuals_NN_train = y_train - y_pred_train
residuals_MLR_train = y_train - y_pred_train_mlr

# Plot outputs
plt.scatter(y_pred_train, residuals_NN_train, s=5, color="blue")
plt.scatter(y_pred_train_mlr, residuals_MLR_train, s=5, color="orange")
plt.legend(["Neural Network" , "Multiple Linear Regression"])
plt.axhline(y=0, color='b')
plt.xlabel('Predicted', fontsize=15)
plt.ylabel('Residuals', fontsize=15)
plt.title('Predicted vs Residuals - Training (6 var)')
plt.show()

# Plot outputs
plt.scatter(y_pred_train, y_train, s=5, color="blue")
plt.scatter(y_pred_train_mlr, y_train, s=5, color="orange")
plt.legend(["Neural Network" , "Multiple Linear Regression"])
plt.plot([min(y_pred_train), max(y_pred_train)], [min(y_train), max(y_train)], color="black", linewidth=2)
plt.xlabel('Predicted', fontsize=15)
plt.ylabel('Observed', fontsize=15)
plt.title('Predicted vs Real Data - Training (75 var)')
plt.show()

# Plot outputs
plt.scatter(y_pred, y_test, s=5, color="blue")
plt.scatter(y_pred_mlr, y_test, s=5, color="orange")
plt.legend(["Neural Network" , "Multiple Linear Regression"])
plt.plot([min(y_pred), max(y_pred)], [min(y_test), max(y_test)], color="black", linewidth=2)
plt.xlabel('Predicted', fontsize=15)
plt.ylabel('Observed', fontsize=15)
plt.title('Predicted vs Real Data - Test (75 var)')
plt.show()

#plot loss functions
plt.plot(H.history['r2_score'])
plt.plot(H.history['val_r2_score'])
plt.title('Neural Network r2_score')
plt.ylabel('r2_score')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
