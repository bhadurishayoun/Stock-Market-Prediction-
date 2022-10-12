#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset_train = pd.read_csv(r"D:\New folder\Google_Stock_Price_Train.csv")
dataset_train.head()


# In[4]:


training_set=dataset_train.iloc[:,1:2].values

print(training_set)
print(training_set.shape)


# In[5]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range =(0,1))
scaled_training_set = scaler.fit_transform(training_set)

scaled_training_set


# In[8]:


x_train =[]
y_train =[]
for i in range (60,1258):
    x_train.append(scaled_training_set[i-60:i,0])
    y_train.append(scaled_training_set[i,0])
x_train=np.array(x_train)
y_train=np.array(y_train)
print (x_train.shape)
print(y_train.shape)


# In[9]:


x_train =np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

x_train.shape


# In[10]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


# In[12]:


regressor =Sequential()
regressor.add(LSTM(units=50, return_sequences=True, input_shape =(x_train.shape[1],1 )))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))


# In[13]:


regressor.compile(optimizer = 'adam', loss ='mean_squared_error')
regressor.fit(x_train, y_train, epochs=100,batch_size=32)


# In[14]:


dataset_test =pd.read_csv(r"D:\New folder\Google_Stock_Price_Test.csv")
actual_stock_price =dataset_test.iloc[:,1:2].values


# In[16]:


dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs =dataset_total[len(dataset_total)- len(dataset_test)-60:].values

inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

x_test =[]
for i in range (60,80):
    x_test.append(inputs[i-60:i,0])
x_test =np.array(x_test)
x_test =np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))


# In[17]:


predicted_stock_price = regressor.predict(x_test)
predicted_stock_price =scaler.inverse_transform(predicted_stock_price)


# In[21]:


plt.plot(actual_stock_price, color ='red', label ='Actual Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()


# In[ ]:




