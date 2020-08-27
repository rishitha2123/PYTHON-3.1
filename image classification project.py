#!/usr/bin/env python
# coding: utf-8

# In[70]:


### IMAGE CLASSIFICATION PROJECT USING MNIST DATASET


# In[71]:


# importing dependencies
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_mldata
get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


# using pandas to read the database stored in the same folder
data = pd.read_csv('mnist_train_100.csv')


# In[73]:


# viewing column heads
data.head()


# In[74]:


# extracting data from dataset and viewing them up close
a = data.iloc[3,1:].values


# In[75]:


# reshaping the extracted data into reasonable size
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[76]:


# preparing the data
# seperating labels and data values
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[77]:


# creating test and train sizes/batches
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4) 


# In[78]:


# check data
x_train.head()


# In[79]:


y_train.head()


# In[80]:


# call rf classifier
rf = RandomForestClassifier(n_estimators=100)


# In[81]:


# fit the model
rf.fit(x_train, y_train)


# In[82]:


# prediction on test data
pred = rf.predict(x_test)


# In[83]:


pred


# In[84]:


# check prediction accuracy
s = y_test.values

# calculate number of correctly predicted values
count = 0
for i in range(len(pred)):
    if pred[i] == s[i]:
        count = count+1


# In[85]:


count


# In[86]:


# total values that the prediction code was run on
len(pred)


# In[87]:


# accuracy value
13/20


# In[88]:


### DONE!!!
### THANK YOU!!!

