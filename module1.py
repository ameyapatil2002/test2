#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data is - qualitative and quantitative
# quantitative is discrete and continuous data


# In[4]:


import sklearn as sk
import seaborn as sns
import pandas as pd
import numpy as np


# In[11]:


#machine learning with steps
from sklearn import tree                     #1
features = [[140,1],[130,1],[150,0],[170,0]] #2  features dalo                         0-bumpy ; 1-shiny    
labels=[0,0,1,1]                             #3  vo feature kiska hai mention karo     0-apples; 1-orange   
clf=tree.DecisionTreeClassifier()            #4 apply library
clf=clf.fit(features, labels)                #5 training
print(clf.predict([[150,1]]))                #6 predict


# In[ ]:


# regression is used for predicting numerical values, linear regression, random forrest regression
# classification is used to classify if pass or fail, true or false, positive or negative, logistic regression, knn
# clustering 
# time value forecasting
# anomaly detection
#






