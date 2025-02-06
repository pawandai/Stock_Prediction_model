#!/usr/bin/env python
# coding: utf-8

# # **1. Import Libraries**

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Dense, Activation

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler


# # **2. Upload Datasets For Stock Data And News Headlines**

# In[ ]:


stock_price = pd.read_csv('ADBL.csv')
stock_headlines = pd.read_csv('raw_news.csv')


# # **3. Data Cleaning**

# In[ ]:


stock_price.head()


# In[ ]:


stock_price.tail()


# In[ ]:


stock_headlines.head()


# In[ ]:


stock_headlines.tail()


# In[ ]:


# displaying number of records in both stock_price and stock_headlines datasets
len(stock_price), len(stock_headlines)


# In[ ]:


# checking for null values in both the datasets
stock_price.isna().any(), stock_headlines.isna().any()


# # **3.1. Numerical Stock Data**

# In[ ]:


# dropping duplicates
stock_price = stock_price.drop_duplicates()

# coverting the datatype of column 'Date' from type object to type 'datetime'
stock_price['published_date'] = pd.to_datetime(stock_price['published_date']).dt.normalize()

# filtering the important columns required
stock_price = stock_price.filter(['published_date', 'close', 'open', 'high', 'low', 'traded_quantity'])

# setting column 'Date' as the index column
stock_price.set_index('published_date', inplace= True)

# sorting the data according to the index i.e 'Date'
stock_price = stock_price.sort_index(ascending=True, axis=0)
stock_price


# In[ ]:


stock_headlines


# # **3.2. Textual News Headlines Data**

# In[ ]:


# dropping duplicates
stock_headlines = stock_headlines.drop_duplicates()

# coverting the datatype of column 'Date' from type string to type 'datetime'
#stock_headlines['published_date'] = stock_headlines['published_date'].astype(str)
#stock_headlines['published_date'] = stock_headlines['published_date'].apply(lambda x: x[0:4]+'-'+x[4:6]+'-'+x[6:8])
#stock_headlines['published_date'] = pd.to_datetime(stock_headlines['published_date'], format='%Y%m%d', errors='coerce').dt.normalize()
stock_headlines['published_date'] = pd.to_datetime(stock_headlines['published_date'], errors='coerce').dt.normalize()

# filtering the important columns required
stock_headlines = stock_headlines.filter(['published_date', 'Title'])

# grouping the news headlines according to 'Date'
stock_headlines = stock_headlines.groupby(['published_date'])['Title'].apply(lambda x: ','.join(x)).reset_index()

# setting column 'Date' as the index column
stock_headlines.set_index('published_date', inplace= True)

# sorting the data according to the index i.e 'Date'
stock_headlines = stock_headlines.sort_index(ascending=True, axis=0)
stock_headlines


# # **4. Combine Stock Data**

# In[ ]:


# concatenating the datasets stock_price and stock_headlines
stock_data = pd.concat([stock_price, stock_headlines], axis=1)

# dropping the null values if any
stock_data.dropna(axis=0, inplace=True)

# displaying the combined stock_data
stock_data


# In[ ]:


#alternate way is to use merge funtion and inner join operation
pd.merge(stock_price, stock_headlines, left_index=True, right_index=True, how='inner')


# # **5. Sentiment Analysis**

# In[ ]:


# adding empty sentiment columns to stock_data for later calculation
stock_data['compound'] = ''
stock_data['negative'] = ''
stock_data['neutral'] = ''
stock_data['positive'] = ''
stock_data.head()


# In[ ]:


import nltk
nltk.download('vader_lexicon')


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata

# instantiating the Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# calculating sentiment scores
stock_data['compound'] = stock_data['Title'].apply(lambda x: sid.polarity_scores(x)['compound'])
stock_data['negative'] = stock_data['Title'].apply(lambda x: sid.polarity_scores(x)['neg'])
stock_data['neutral'] = stock_data['Title'].apply(lambda x: sid.polarity_scores(x)['neu'])
stock_data['positive'] = stock_data['Title'].apply(lambda x: sid.polarity_scores(x)['pos'])

# displaying the stock data
stock_data.head()


# In[ ]:


# dropping the 'headline_text' which is unwanted now
stock_data.drop(['Title'], inplace=True, axis=1)

# rearranging the columns of the whole stock_data
stock_data = stock_data[['close', 'compound', 'negative', 'neutral', 'positive', 'open', 'high', 'low', 'traded_quantity']]

# set the index name
stock_data.index.name = 'Date'

# displaying the final stock_data
stock_data.head()


# In[ ]:


# writing the prepared stock_data to disk
stock_data.to_csv('stock_data.csv')


# # **6. Exploratory Data Analysis**

# In[ ]:


# displaying the shape i.e. number of rows and columns of stock_data
stock_data.shape


# In[ ]:


# checking for null values
stock_data.isna().any()


# In[ ]:


# displaying stock_data statistics
stock_data.describe(include='all')


# In[ ]:


# displaying stock_data information
stock_data.info()


# In[ ]:


# setting figure size
plt.figure(figsize=(20,10))

# plotting close price
stock_data['close'].plot()

# setting plot title, x and y labels
plt.title("Close Price")
plt.xlabel('Date')
plt.ylabel('Close Price ($)')


# In[ ]:


# calculating 7 day rolling mean
stock_data.rolling(7).mean().head(20)


# In[ ]:


# setting figure size
plt.figure(figsize=(16,10))

# plotting the close price and a 30-day rolling mean of close price
stock_data['close'].plot()
stock_data.rolling(window=30).mean()['close'].plot()


# # **7. Data Preparation**

# In[ ]:


# calculating data_to_use
percentage_of_data = 1.0
data_to_use = int(percentage_of_data*(len(stock_data)-1))

# using 80% of data for training
train_end = int(data_to_use*0.8)
total_data = len(stock_data)
start = total_data - data_to_use

# printing number of records in the training and test datasets
print("Number of records in Training Data:", train_end)
print("Number of records in Test Data:", total_data - train_end)


# In[ ]:


# predicting one step ahead
steps_to_predict = 1

# capturing data to be used for each column
close_price = stock_data.iloc[start:total_data,0] #close
compound = stock_data.iloc[start:total_data,1] #compound
negative = stock_data.iloc[start:total_data,2] #neg
neutral = stock_data.iloc[start:total_data,3] #neu
positive = stock_data.iloc[start:total_data,4] #pos
open_price = stock_data.iloc[start:total_data,5] #open
high = stock_data.iloc[start:total_data,6] #high
low = stock_data.iloc[start:total_data,7] #low
volume = stock_data.iloc[start:total_data,8] #volume

# printing close price
print("Close Price:")
close_price


# In[ ]:


# shifting next day close
close_price_shifted = close_price.shift(-1)

# shifting next day compound
compound_shifted = compound.shift(-1)

# concatenating the captured training data into a dataframe
data = pd.concat([close_price, close_price_shifted, compound, compound_shifted, volume, open_price, high, low], axis=1)

# setting column names of the revised stock data
data.columns = ['close_price', 'close_price_shifted', 'compound', 'compound_shifted','volume', 'open_price', 'high', 'low']

# dropping nulls
data = data.dropna()
data.head(10)


# # **7.1. Setting Target Variable And Feature Dataset**

# In[ ]:


# setting the target variable as the shifted close_price
y = data['close_price_shifted']
y


# In[ ]:


# setting the features dataset for prediction
cols = ['close_price', 'compound', 'compound_shifted', 'volume', 'open_price', 'high', 'low']
x = data[cols]
x


# # **7.3. Scaling the Target Variable and the Feature Dataset**
# 
# Since we are using LSTM to predict stock prices, which is a time series data, it is important to understand that LSTM can be very sensitive to the scale of the data. Right now, if the data is observed, it is present in different scales. Therefore, it is important to re-scale the data so that the range of the dataset is same, for almost all records. Here a feature range of (-1,1) is used

# In[ ]:


# scaling the feature dataset
scaler_x = preprocessing.MinMaxScaler (feature_range=(-1, 1))
x = np.array(x).reshape((len(x) ,len(cols)))
x = scaler_x.fit_transform(x)

# scaling the target variable
scaler_y = preprocessing.MinMaxScaler (feature_range=(-1, 1))
y = np.array (y).reshape ((len( y), 1))
y = scaler_y.fit_transform (y)

# displaying the scaled feature dataset and the target variable
x, y


# # **7.4. Dividing the dataset into Training and Test**
# 
# Normally for any other dataset train_test_split from sklearn package is used, but for time series data like stock prices which is dependent on date, the dataset is divided into train and test dataset in a different way as shown below. In timeseries data, an observation for a particular date is always dependent on the previous date records.

# In[ ]:


# preparing training and test dataset
X_train = x[0 : train_end,]
X_test = x[train_end+1 : len(x),]
y_train = y[0 : train_end]
y_test = y[train_end+1 : len(y)]

# printing the shape of the training and the test datasets
print('Number of rows and columns in the Training set X:', X_train.shape, 'and y:', y_train.shape)
print('Number of rows and columns in the Test set X:', X_test.shape, 'and y:', y_test.shape)


# In[ ]:


# reshaping the feature dataset for feeding into the model
X_train = X_train.reshape (X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

# printing the re-shaped feature dataset
print('Shape of Training set X:', X_train.shape)
print('Shape of Test set X:', X_test.shape)


# # **9. Stock Data Modelling**

# In[ ]:


# setting the seed to achieve consistent and less random predictions at each execution
np.random.seed(2016)

# setting the model architecture
model=Sequential()
model.add(LSTM(100,return_sequences=True,activation='tanh',input_shape=(len(cols),1)))
model.add(Dropout(0.1))
model.add(LSTM(100,return_sequences=True,activation='tanh'))
model.add(Dropout(0.1))
model.add(LSTM(100,activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(1))

# printing the model summary
model.summary()


# In[ ]:


# compiling the model
model.compile(loss='mse' , optimizer='adam')

# fitting the model using the training dataset
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=8, verbose=1)


# # **9.1. Saving the Model to disk**

# In[ ]:


# saving the model as a json file
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights('model.weights.h5') # Change the filename to include .weights.h5
print('Model is saved to the disk')


# # **10. Model Predictions**

# In[ ]:


# performing predictions
predictions = model.predict(X_test)

# unscaling the predictions
predictions = scaler_y.inverse_transform(np.array(predictions).reshape((len(predictions), 1)))

# printing the predictions
print('Predictions:')
predictions[0:5]


# # **11. Model Evaluation**

# In[ ]:


# calculating the training mean-squared-error
train_loss = model.evaluate(X_train, y_train, batch_size = 1)

# calculating the test mean-squared-error
test_loss = model.evaluate(X_test, y_test, batch_size = 1)

# printing the training and the test mean-squared-errors
print('Train Loss =', round(train_loss,4))
print('Test Loss =', round(test_loss,4))


# In[ ]:


# calculating root mean squared error
root_mean_square_error = np.sqrt(np.mean(np.power((y_test - predictions),2)))
print('Root Mean Square Error =', round(root_mean_square_error,4))


# In[ ]:


# calculating root mean squared error using sklearn.metrics package
rmse = metrics.mean_squared_error(y_test, predictions)
print('Root Mean Square Error (sklearn.metrics) =', round(np.sqrt(rmse),4))


# # **12. Plotting the Predictions against unseen data**

# In[ ]:


# unscaling the test feature dataset, x_test
X_test = scaler_x.inverse_transform(np.array(X_test).reshape((len(X_test), len(cols))))

# unscaling the test y dataset, y_test
y_train = scaler_y.inverse_transform(np.array(y_train).reshape((len(y_train), 1)))
y_test = scaler_y.inverse_transform(np.array(y_test).reshape((len(y_test), 1)))


# In[ ]:


# plotting
plt.figure(figsize=(16,10))

# plt.plot([row[0] for row in y_train], label="Training Close Price")
plt.plot(predictions, label="Predicted Close Price")
plt.plot([row[0] for row in y_test], label="Testing Close Price")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
plt.show()
