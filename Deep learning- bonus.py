#!/usr/bin/env python
# coding: utf-8

# In[67]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

SEED = 123 
DATA_SPLIT_PCT = 0.2

LABELS = ["Normal","Attack"]


# In[68]:


df_train1 = pd.read_csv('batadal_train_1.csv', ',')
df_train2 = pd.read_csv('batadal_train_2.csv', ',')
df_test_init = pd.read_csv('batadal_test.csv', ',')

#Remove leading and trailing whitespaces in column names
df_train1.rename(columns=lambda x: x.strip(), inplace = 'true')
df_train2.rename(columns=lambda x: x.strip(), inplace = 'true')
df_test_init.rename(columns=lambda x: x.strip(), inplace = 'true')

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
df_test_init['ATT_FLAG'] = df_test_init['L_T1']
for index, row in df_test_init.iterrows():
    if row['DATETIME'] in {'16/01/17 09', '30/01/17 08', '09/02/17 03', '12/02/17 01', '24/02/17 05', '10/03/17 14', '25/03/17 20'}:
        set_test = 1
    if row['DATETIME'] in {'19/01/17 06', '02/02/17 00', '10/02/17 09', '13/02/17 07', '28/02/17 08', '13/03/17 21', '27/03/17 01'}:
        set_test = 0
    if(set_test ==1): 
        df_test_init.at[index, 'ATT_FLAG'] = 1
    else:
        df_test_init.at[index, 'ATT_FLAG'] = 0
        
#Merge the 2 training datasets
df_train = pd.concat([df_train1, df_train2])

#Drop columns as they contain no info
df_train = df_train.drop([ 'DATETIME', 'S_PU5', 'F_PU5', 'S_PU9', 'F_PU9'], axis = 1)
df_test = df_test_init.drop(['DATETIME','S_PU5', 'F_PU5', 'S_PU9', 'F_PU9'], axis = 1)

df = df_train


# Splitting into training, validation, group into classes

# In[69]:


df_train, df_valid = train_test_split(df_train, test_size=DATA_SPLIT_PCT, random_state=SEED)


# In[70]:


df_train_0 = df_train.loc[df_train['ATT_FLAG'] == 0]
df_train_1 = df_train.loc[df_train['ATT_FLAG'] == 1]

df_train_0_x = df_train_0.drop(['ATT_FLAG'], axis=1)
df_train_1_x = df_train_1.drop(['ATT_FLAG'], axis=1)


df_valid_0 = df_valid.loc[df_valid['ATT_FLAG'] == 0]
df_valid_1 = df_valid.loc[df_valid['ATT_FLAG'] == 1]

df_valid_0_x = df_valid_0.drop(['ATT_FLAG'], axis=1)
df_valid_1_x = df_valid_1.drop(['ATT_FLAG'], axis=1)


df_test_0 = df_test.loc[df_test['ATT_FLAG'] == 0]
df_test_1 = df_test.loc[df_test['ATT_FLAG'] == 1]

df_test_0_x = df_test_0.drop(['ATT_FLAG'], axis=1)
df_test_1_x = df_test_1.drop(['ATT_FLAG'], axis=1)


# Scale the data to 0-1 range

# In[71]:


scaler = StandardScaler().fit(df_train_0_x)
df_train_0_x_rescaled = scaler.transform(df_train_0_x)
df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
df_valid_x_rescaled = scaler.transform(df_valid.drop(['ATT_FLAG'], axis = 1))

df_test_0_x_rescaled = scaler.transform(df_test_0_x)
df_test_x_rescaled = scaler.transform(df_test.drop(['ATT_FLAG'], axis = 1))


# ## Set up a neural network with relu-encoder

# In[72]:


nb_epoch = 100
batch_size = 128
input_dim = df_train_0_x_rescaled.shape[1] #num of predictor variables, 
encoding_dim = 16
hidden_dim = int(encoding_dim / 2)
learning_rate = 1e-1

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


# This requires pydot and graphviz installed

# In[73]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(autoencoder, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))


# ## Train the network

# In[74]:


autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer='adam')

cp = ModelCheckpoint(filepath="autoencoder_classifier.h5",
                               save_best_only=True,
                               verbose=0)

tb = TensorBoard(log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=True)

history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
                    verbose=1,
                    callbacks=[cp, tb]).history


# ## Validate 

# In[75]:


valid_x_predictions = autoencoder.predict(df_valid_x_rescaled)
mse = np.mean(np.power(df_valid_x_rescaled - valid_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': df_valid['ATT_FLAG'] == 1})

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold / reconstruction error')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()


# In[76]:


threshold_fixed = 1.5

test_x_predictions = autoencoder.predict(df_test_x_rescaled)
mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)
error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': df_test['ATT_FLAG']})
error_df_test = error_df_test.reset_index()

groups = error_df_test.groupby('True_class')

fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Attack" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();


# In[77]:


# Convert string to datetime for plotting purposes - RUN ONCE ONLY
from datetime import datetime
def string_to_datetime(date_string):#convert time string datetime 
    return datetime.strptime(date_string, '%d/%m/%y %H')

df_test_init['DATETIME'] = df_test_init['DATETIME'].apply(string_to_datetime)


# In[78]:


th = 1.9
pred_y_t = [1 if e > th else 0 for e in error_df_test.Reconstruction_error.values]
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
    
dates = list(df_test_init['DATETIME'])
flags = list(df_test_init['ATT_FLAG'])


plot_series(dates, flags, pred_y_t, 'predictions')

conf_matrix = confusion_matrix(error_df_test.True_class, pred_y_t)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# Plot the ROC curve for the chosen threshold

# In[79]:


false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df_test.True_class, error_df_test.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




