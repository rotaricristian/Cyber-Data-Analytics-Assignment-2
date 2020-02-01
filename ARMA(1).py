#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


train = pd.read_csv('BATADAL/BATADAL_dataset04.csv', ',')
test = pd.read_csv('BATADAL/BATADAL_test_dataset.csv', ',')

#Remove leading and trailing whitespaces in column names
train.rename(columns=lambda x: x.strip(), inplace = 'true')
test.rename(columns=lambda x: x.strip(), inplace = 'true')
    

#Label the training data
set_train =0;
for index, row in train.iterrows():
    if row['DATETIME'] in {'13/09/16 23', '26/09/16 11', '09/10/16 09', '29/10/16 19', '26/11/16 17', '06/12/16 07', '14/12/16 15'}:
        set_train = 1
    if row['DATETIME'] in {'16/09/16 00', '27/09/16 10', '11/10/16 20', '02/11/16 16', '29/11/16 04', '10/12/16 04', '19/12/16 04'}:
        set_train = 0
    if(set_train ==1): 
        train.set_value(index, 'ATT_FLAG', 1)
    else:
        train.set_value(index, 'ATT_FLAG', 0)
        
#Label the testing data
set_test = 0
test['ATT_FLAG'] = train['ATT_FLAG']
for index, row in test.iterrows():
    if row['DATETIME'] in {'16/01/17 09', '30/01/17 08', '09/02/17 03', '12/02/17 01', '24/02/17 05', '10/03/17 14', '25/03/17 20'}:
        set_test = 1
    if row['DATETIME'] in {'19/01/17 06', '02/02/17 00', '10/02/17 09', '13/02/17 07', '28/02/17 08', '13/03/17 21', '27/03/17 01'}:
        set_test = 0
    if(set_test ==1): 
        test.set_value(index, 'ATT_FLAG', 1)
    else:
        test.set_value(index, 'ATT_FLAG', 0)
        
#drop columns as they contain no info
train = train.drop(['S_PU3', 'F_PU3', 'S_PU5', 'F_PU5', 'S_PU9', 'F_PU9'], axis = 1)
test = test.drop(['S_PU3', 'F_PU3', 'S_PU5', 'F_PU5', 'S_PU9', 'F_PU9'], axis = 1)
        
print(train.columns.values)


# In[4]:


from datetime import datetime

def string_to_datetime(date_string):#convert time string datetime 
    return datetime.strptime(date_string, '%d/%m/%y %H')#.timestamp()

train['DATETIME'] = train['DATETIME'].apply(string_to_datetime)
test['DATETIME'] = test['DATETIME'].apply(string_to_datetime)


# In[5]:


Y_train = train['ATT_FLAG'].values
train = train.drop('ATT_FLAG',  axis = 1)
X_train = train.values.tolist()

Y_test = test['ATT_FLAG'].values
test = test.drop('ATT_FLAG',  axis = 1)
X_test = test.values.tolist()

X_test = np.asarray(X_test)
X_train = np.asarray(X_train)


# In[6]:


all_cols_series_train = []
all_cols_series_test = []

for i in range(1, X_train.shape[1]):
    s = pd.Series(X_train[:, i].astype(float), index=X_train[:, 0])
    all_cols_series_train.append(s)
    
for i in range(1, X_test.shape[1]):
    s = pd.Series(X_test[:, i].astype(float), index=X_test[:, 0])
    all_cols_series_test.append(s)


# In[7]:


from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

max_lags = 80

cols = list(train.columns.values)
for i in range(5):
    print(cols[i+1]) 
    plot_acf(all_cols_series_train[i], lags=max_lags)
    pyplot.show()
    
for i in range(5):
    print(cols[i+1]) 
    plot_pacf(all_cols_series_train[i], lags=max_lags)
    pyplot.show()


# In[8]:


from sklearn.metrics import mean_squared_error

def evaluate_model(series, p, q): 
    size = int(len(series) * 0.66)
    train, test = series[0:size], series[size:len(series)]
    model = ARIMA(train, order=(p, 0, q)) 
    model_fit = model.fit(disp=0) 
    pred = model_fit.forecast(len(test))[0]
    error = mean_squared_error(test, pred)
    aic = model_fit.aic
    return error, aic


# In[9]:


def evaluate_models(dataset, p_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for q in q_values:
            try:
                mse, aic = evaluate_model(dataset, p, q)
                if aic < best_score:
                    best_score, best_cfg = aic, (p,q)
                    print('ARIMA%s MSE=%.3f AIC=%.3f' % ((p,q),mse, aic))
            except:
                continue


# In[10]:


# ARIMA is EXTREMELY slow, so it will take time!

import warnings
warnings.filterwarnings("ignore")

p_vals = [1,2,3,5,10]
q_vals = [1,2,3,5,10]

sensor_indexes = [x for x in range(5)]

for index in sensor_indexes:
    print('Sensor: ', index+1)
    evaluate_models(all_cols_series_train[index], p_vals, q_vals)


# In[48]:


theshold_times_std = 2.5
thresholds = []

def analyse_model(model):
    model_fit = model.fit()
    
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    th = abs(residuals.mean()[0]) + theshold_times_std*residuals.std()[0]
    thresholds.append(th)
    print(residuals.describe())
    
def analyse_res(model):
    model_fit = model.fit()
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()


# In[49]:


def plot_series(dates, pl1, pl2, name):
    fig, ax1 = pyplot.subplots()
    ax1.plot(dates, pl1, 'r-')
    ax1.set_xlabel('date')
    ax1.set_ylabel('Attack', color='red')
    ax1.tick_params('y', colors='red')

    ax2 = ax1.twinx()
    ax2.plot(dates, pl2, 'b--')
    ax2.set_ylabel(name, color='b')
    ax2.tick_params('y', colors='b')

    fig.tight_layout()
    pyplot.show()


# In[50]:


analyse_model(ARIMA(all_cols_series_train[0], order=(10,0,10)))


# In[51]:


analyse_model(ARIMA(all_cols_series_train[3], order=(5,0,2)))


# In[52]:


analyse_model(ARIMA(all_cols_series_train[4], order=(5,0,2)))


# In[53]:


def metric(tags, labels):
    TP, FP = 0, 0
    for t, l in zip(tags, labels):
        if t == 1 and l == 1: TP+=1
        if t == 0 and l == 1: FP+=1
    return TP, FP


# In[54]:


def predict_anomalies(model, threshold):
    model_fit = model.fit()
    residuals = pd.DataFrame(model_fit.resid)
    pred = []
    for res in residuals.values:
        if abs(res)>threshold:
            pred.append(1)
        else:
            pred.append(0)
    return pred

l = [0, 3, 4]

for th, i in zip(thresholds, l):
    model = ARIMA(all_cols_series_test[i], order=(5, 0, 2))
    analyse_res(model)
    pred = predict_anomalies(model, th)
    print(metric(Y_test, pred))
    plot_series(X_test[:, 0], Y_test, pred, 'Chart: ' + str(i))


# In[ ]:




