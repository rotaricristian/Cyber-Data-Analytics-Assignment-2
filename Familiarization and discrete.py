#!/usr/bin/env python
# coding: utf-8

# # Anomaly detection

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from nltk import trigrams
from nltk.util import ngrams
import nltk


# In[16]:


df_train1 = pd.read_csv('batadal_train_1.csv', ',')
df_train2 = pd.read_csv('batadal_train_2.csv', ',')
df_test = pd.read_csv('batadal_test.csv', ',')

#Remove leading and trailing whitespaces in column names
df_train1.rename(columns=lambda x: x.strip(), inplace = 'true')
df_train2.rename(columns=lambda x: x.strip(), inplace = 'true')
df_test.rename(columns=lambda x: x.strip(), inplace = 'true')

#Label the training data 2
set_train =0;
for index, row in df_train2.iterrows():
    if row['DATETIME'] in {'13/09/16 23', '26/09/16 11', '09/10/16 09', '29/10/16 19', '26/11/16 17', '06/12/16 07', '14/12/16 15'}:
        set_train = 1
    if row['DATETIME'] in {'16/09/16 00', '27/09/16 10', '11/10/16 20', '02/11/16 16', '29/11/16 04', '10/12/16 04', '19/12/16 04'}:
        set_train = 0
    if(set_train ==1): 
        df_train2.at[index, 'ATT_FLAG'] = 1
    else:
        df_train2.at[index, 'ATT_FLAG'] = 0
        
#Label the testing data
set_test = 0
df_test['ATT_FLAG'] = df_test['L_T1']
for index, row in df_test.iterrows():
    if row['DATETIME'] in {'16/01/17 09', '30/01/17 08', '09/02/17 03', '12/02/17 01', '24/02/17 05', '10/03/17 14', '25/03/17 20'}:
        set_test = 1
    if row['DATETIME'] in {'19/01/17 06', '02/02/17 00', '10/02/17 09', '13/02/17 07', '28/02/17 08', '13/03/17 21', '27/03/17 01'}:
        set_test = 0
    if(set_test ==1): 
        df_test.at[index, 'ATT_FLAG'] = 1
    else:
        df_test.at[index, 'ATT_FLAG'] = 0
        
#Merge the 2 training datasets
df_train = pd.concat([df_train1, df_train2])

#Drop columns as they contain no info
df_train = df_train.drop([ 'S_PU5', 'F_PU5', 'S_PU9', 'F_PU9'], axis = 1)
df_test = df_test.drop(['S_PU5', 'F_PU5', 'S_PU9', 'F_PU9'], axis = 1)
        

print(df_train.columns.values)


# In[17]:


#convert string to datetime for plotting purposes
def string_to_datetime(date_string):#convert time string datetime 
    return datetime.strptime(date_string, '%d/%m/%y %H')

df_train2['DATETIME'] = df_train2['DATETIME'].apply(string_to_datetime)
df_train['DATETIME'] = df_train['DATETIME'].apply(string_to_datetime)
df_test['DATETIME'] = df_test['DATETIME'].apply(string_to_datetime)


# In[18]:


def plot_series(dates, pl1, pl2, name):
    fig, ax1 = plt.subplots()
    ax1.plot(dates, pl1, 'r-')
    ax1.set_xlabel('date')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Attack', color='red')
    ax1.tick_params('y', colors='red')

    ax2 = ax1.twinx()
    ax2.plot(dates, pl2, 'b--')
    ax2.set_ylabel(name, color='b')
    ax2.tick_params('y', colors='b')

    fig.tight_layout()
    plt.show()


# In[19]:


#Rolling average windows
df_rolling = df_test.rolling(5, on='DATETIME', min_periods = 1).mean()

#Prediction for L_T2
error=[]
avg = list(df_rolling['L_T2'].values)
real = list(df_test[ 'L_T2'].values)    
total=0
for  a, r  in zip(avg, real):
    val = round(100*abs(a - r)/r)
    error.append(val)
    total+=val
    
print(total/len(avg))
dates = list(df_test['DATETIME'])
flags = list(df_test['ATT_FLAG'])


plot_series(dates, flags, error, 'T3 prediction error in %')


# In[20]:


def metric(tags, labels):
    TP, FP = 0, 0
    for t, l in zip(tags, labels):
        if t == 1 and l == 1: TP+=1
        if t == 0 and l == 1: FP+=1
    return TP, FP


# In[21]:


discrete_train2 = pd.cut(df_train2['P_J307'].values, 5, labels = False)

dates = list(df_train2['DATETIME'])
flags = list(df_train2['ATT_FLAG'])


plot_series(dates, flags, discrete_train2, 'Discretized P_J307 data')


# In[22]:


#discretize the data into equal bins and apply N-grams to it
def discretize_ngram(bins, n_grams, train, test, threshold):
    discrete_train = pd.cut(train, bins, labels = False)
    discrete_test = pd.cut(test, bins, labels = False)
    
    tokens_train = ngrams(discrete_train, n_grams)
    tokens_test = ngrams(discrete_test, n_grams)
    
    fdist = nltk.FreqDist(tokens_train)
    labels=[0]*(n_grams - 1)
    
    for i in tokens_test:
        if fdist.freq(i) < threshold:
            labels.append(1)
        else:
            labels.append(0)
         
    return labels


# In[12]:


#model each sensor with Ngrams and plot the results 

dates = list(df_test['DATETIME'])
flags = list(df_test['ATT_FLAG'])

final=[]

F_V2 = discretize_ngram(5, 4, df_train['F_V2'].values, df_test['F_V2'].values, 0.00002 ) 
plot_series(dates, flags, F_V2, 'F_V2')
print(metric(flags, F_V2))

L_T2 = discretize_ngram(5, 3, df_train['L_T2'].values, df_test['L_T2'].values, 0.0005 ) # for 4,3 - 0.00002- (9,3)
plot_series(dates, flags, L_T2, 'L_T2')
print(metric(flags, L_T2))

L_T3 = discretize_ngram(4, 3, df_train['L_T3'].values, df_test['L_T3'].values, 0.0025 ) # or 4,3 - 0.0025 - (6,2)
plot_series(dates, flags, L_T3, 'L_T3')
print(metric(flags, L_T3))


P_J422 = discretize_ngram(4, 3, df_train['P_J422'].values, df_test['P_J422'].values, 0.0025 ) #for 4,3 - 0.0025 (6,2)
plot_series(dates, flags, P_J422, 'P_J422')
print(metric(flags, P_J422))

for a,b,c,d in zip(F_V2, L_T2, L_T3, P_J422):
    if (a+b+c+d)>0: final.append(1)
    else: final.append(0)

plot_series(dates, flags, final, 'discrete')



    


# In[ ]:




