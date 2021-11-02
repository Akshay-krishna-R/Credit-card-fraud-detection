#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[4]:


data_set = pd.read_csv(r"creditcard.csv")


# In[6]:


data_set.shape


# In[7]:


data_set.head()


# In[10]:


data_set["Class"].unique()


# In[11]:


data_set["Class"].nunique()


# In[12]:


data_set["Class"].value_counts()


# In[14]:


data_set.isnull().sum()


# In[13]:


data_set.info()


# In[15]:


data_set.describe()


# In[16]:


correlation_dataset = data_set.corr()


# In[18]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation_dataset, cbar = True, square=True, fmt='.1g', annot= True,  annot_kws = {'size':8}, cmap='Greens')


# #### Finding highly correlated independent features 

# In[26]:


def correaltion_function(dataset,threshold):
    col_corr = set()
    correlation = dataset.corr()
    for i in range(len(correlation)):
        for j in range(i):
            if abs(correlation.iloc[i,j]) > threshold:
                col_name = correlation.columns[i]
                col_corr.add(col_name)
    
    return col_corr


# In[27]:


correalted_features_dup = correaltion_function(data_set,0.9)
len(correalted_features_dup)


# In[28]:


data_set.head()


# In[33]:


x= data_set.drop("Class",axis = 1)
y = data_set["Class"]


# In[35]:


x.shape


# In[36]:


y.shape


# In[37]:


x.head()


# In[38]:


y.head()


# In[46]:


X_train, X_test, Y_train, Y_test  =  train_test_split(x, y, test_size=0.10, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[47]:


f,ax = plt.subplots(figsize=(10,8))
sns.scatterplot(x="Time",y="Class",data= data_set,label = "Class vs Time")
sns.set(style="ticks")
sns.set(context="paper")
ax.legend(loc="upper right",frameon=True)


# In[ ]:





# In[48]:


sclaer = StandardScaler()
sclaer.fit(X_train)
X_train = sclaer.transform(X_train)
X_test = sclaer.transform(X_test)


# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


# In[50]:


model = LogisticRegression()
model.fit(X_train, Y_train)


# In[52]:


Y_pred_test = model.predict(X_test)
Y_pred_test_accuracy = accuracy_score(Y_test, Y_pred_test)
print("Accuracy on testing data : ", Y_pred_test_accuracy)


# In[57]:


Y_pred_test


# In[72]:


data_to_check = [-0.56706426, -0.32819847, -0.03492045,  0.98450497, -1.19223017,
       -0.83323377,  0.19466551, -1.12280862, -1.95713547,  1.06376516,
       -1.91675211,  0.46979084,  0.47406607, -2.20235832,  0.80814307,
        0.3225051 ,  0.46316261, -0.63901139,  1.23092288,  0.0213219 ,
        0.80136542, -1.67266914,  0.35520116, -0.59398706, -0.06408794,
        2.68220711, -1.37976661,  0.07786754,  0.5470093 ,  0.0459251 ]


# In[76]:


transaction_detection(data_to_check)


# In[75]:


def transaction_detection(X_check):
    #np.asarray(X_check)
    X_check = np.reshape(X_check,(1,-1))

    Y_result = model.predict(X_check)
    if Y_result == 0:
        print("Its a fraud transcation")
    else:
        print("Its a legal transcation")


# In[ ]:





# In[ ]:





# In[ ]:





# In[86]:


x= data_set.drop("Class",axis = 1)
y = data_set["Class"]


# In[87]:


from imblearn.over_sampling import SMOTE


# In[109]:


X_train, X_test, Y_train, Y_test  =  train_test_split(x, y, test_size=0.6, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[110]:


from collections import Counter
counter = Counter(Y_train)
counter


# In[111]:


smt = SMOTE()
X_train_sm,Y_train_sm = smt.fit_resample(X_train,Y_train)


# In[112]:


counter = Counter(Y_train_sm)
counter


# In[113]:


count_classes = Y_train_sm.value_counts()
count_classes.plot(kind="bar",rot=0)
plt.title("Class distribution")
plt.xlabel("Classes")
plt.ylabel("Frequency")


# In[114]:


X_train_sm.shape,Y_train_sm.shape


# In[115]:


model = LogisticRegression()
model.fit(X_train_sm, Y_train_sm)


# In[117]:


Y_pred_test = model.predict(X_test)
Y_pred_test_accuracy = accuracy_score(Y_test, Y_pred_test)
print("Accuracy on testing data : ", Y_pred_test_accuracy)


# In[139]:


def transaction_detection(X_check):
    #np.asarray(X_check)
    X_check = np.reshape(X_check,(1,-1))

    Y_result = model.predict(X_check)
    if Y_result == 0:
        print("Its a legal transcation")
    else:
        print("Its a fraud transcation")


# In[144]:





# In[ ]:





# In[145]:


transaction_detection(data_to_check)

