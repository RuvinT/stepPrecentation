#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 22:58:43 2023

@author: ruvinjagoda
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import math
from keras.layers import Dropout
import seaborn as sb
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier



df = pd.read_csv('AAPL.csv')

# Convert the 'Date' column to a datetime object
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)

# Filter the rows based on the 'Date' column
df = df[df['Date'].dt.year >= 2016]
 
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
data_grouped = df.groupby('year').mean()

'''
plt.subplots(figsize=(20,10))
 
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()
df = df.drop('Date', axis=1)
'''


df.groupby('is_quarter_end').mean()


df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

'''
plt.pie(df['target'].value_counts().values,
        labels=[0, 1], autopct='%1.1f%%')
plt.show()

plt.figure(figsize=(10, 10))
 
# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
'''
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']


scaler = StandardScaler()
features = scaler.fit_transform(features)


 
X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

print(Y_train)
num_timesteps = 3
X_train = X_train.reshape((X_train.shape[0], num_timesteps, X_train.shape[1] // num_timesteps))
X_valid = X_valid.reshape((X_valid.shape[0], num_timesteps, X_valid.shape[1] // num_timesteps))  
    
# define model architecture
model = Sequential([
    LSTM(64, activation='relu', input_shape=X_train.shape[1:]),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(X_train, Y_train, epochs=200, batch_size=32, validation_data=(X_valid, Y_valid))
       

# Evaluate model accuracy on test set
loss, accuracy = model.evaluate(X_valid, Y_valid, verbose=0)
print('Validation accuracy:', accuracy)

# Get training accuracy from history object
train_accuracy = history.history['accuracy'][-1]
print('Training accuracy:', train_accuracy)

# Save trained model to file
model.save('my_model.h5')

import joblib

joblib.dump(scaler, 'scaler.pkl')

# Get the models predicted price values 


data = {'Date': ['2017-03-14 00:00:00'], 
        'Low': [34.709999],
        'Open': [34.825001],
        'Volume': [61236400],
        'High': [34.912498],
        'Close': [34.747501]}

dfp = pd.DataFrame(data)


dfp['open-close']  = dfp['Open'] - dfp['Close']
dfp['low-high']  = dfp['Low'] - dfp['High']

splitted = dfp['Date'].str.split('-', expand=True)
 
dfp['month'] = splitted[1].astype('int')

dfp['is_quarter_end'] = np.where(dfp['month']%3==0,1,0)


x_test= dfp[['open-close', 'low-high', 'is_quarter_end']]
new_data = scaler.transform(x_test)  # Scale the data using the scaler object used for training

# Reshape the data into the expected shape
new_data = new_data.reshape((1, num_timesteps, X_train.shape[1] // num_timesteps))

# Make a prediction
prediction = model.predict(new_data)

# The prediction is a probability value between 0 and 1, so you can round it to get a binary classification result
binary_prediction = round(prediction[0][0])

print('Probability:', prediction[0][0])
print('Binary prediction:', binary_prediction)


'''
# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(rmse) 

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
train_d = df[ 'Date'][:training_data_len]
valid_d = df[ 'Date'][training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_d,train['Close'])
plt.plot(valid_d,valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show() 
'''