#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot


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

def string_to_datetime(date_string):
    return datetime.strptime(date_string, '%d/%m/%y %H').timestamp()

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
print(X_train.shape)


# In[6]:


X_train_ = X_train[:, 1:]
X_test_ = X_test[:, 1:]
print(X_train_.shape)


# In[7]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=[0, 1])
X_train_ = scaler.fit_transform(X_train_) 
X_test_ = scaler.fit_transform(X_test_) 


# In[8]:


def calculate_anomaly_scores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss


# In[9]:


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


# In[10]:


def plot_varience(pca):
    pyplot.figure()
    pyplot.plot(np.cumsum(pca.explained_variance_ratio_))
    pyplot.xlabel('Number of Components')
    pyplot.ylabel('Variance (%)')
    pyplot.title('Dataset Variance')
    pyplot.show()


# In[11]:


from sklearn.decomposition import PCA

pca = PCA(random_state=42)
X_train_PCA = pca.fit_transform(X_train_)
plot_varience(pca)


# In[12]:


pca = PCA(random_state=42, n_components=13)

X_train_PCA = pca.fit_transform(X_train_)

X_train_PCA = pd.DataFrame(data=X_train_PCA, index=X_train[:,0])

X_train_PCA_inverse = pca.inverse_transform(X_train_PCA)
X_train_PCA_inverse = pd.DataFrame(data=X_train_PCA_inverse, index=X_train[:,0])


# In[13]:


scores = calculate_anomaly_scores(pd.DataFrame(X_train_), X_train_PCA_inverse)

plot_series(train['DATETIME'], Y_train, scores, 'Scores')


# In[14]:


threshold = 0.1

preds = [1 if score > threshold else 0 for score in scores]

TP = TN = FP = FN = 0
for pred,label in zip(preds, Y_train):
    
    if label==1:
        if label == pred:
            TP+=1
        else:
            FN+=1
    else:
        if label == pred:
            TN+=1
        else:
            FP+=1
        
print(FP, TP, TN, FN)    


# In[15]:


plot_series(train['DATETIME'], Y_train, preds, 'Predictions')


# In[16]:


residuals = X_train_-X_train_PCA_inverse

r = residuals[10]
r.plot()
pyplot.show()
r.plot(kind='kde')
pyplot.show()


# In[17]:


# Train on training data and run it on test data for comparison purposes
pca = PCA(random_state=42, n_components=13)

X_train_PCA = pca.fit(X_train_)

X_test_trans = pca.transform(X_test_)

X_test_PCA_inverse = pca.inverse_transform(X_test_trans)
X_test_PCA_inverse = pd.DataFrame(data=X_test_PCA_inverse, index=X_test[:,0])


# In[18]:


scores = calculate_anomaly_scores(pd.DataFrame(X_test_), X_test_PCA_inverse)


# In[19]:


threshold = 0.6

preds = [1 if score > threshold else 0 for score in scores]

TP = TN = FP = FN = 0
for pred,label in zip(preds, Y_test):
    
    if label==1:
        if label == pred:
            TP+=1
        else:
            FN+=1
    else:
        if label == pred:
            TN+=1
        else:
            FP+=1
        
print(FP, TP, TN, FN)    


# In[20]:


plot_series(test['DATETIME'], Y_test, preds, 'Predictions')


# In[ ]:




