#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv("apple_quality.csv")
df.head()


# In[3]:


df.shape


# In[4]:



df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df = df.dropna()


# In[7]:


df.isnull().sum()


# In[8]:


df['Quality'].unique()    


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Quality'] = encoder.fit_transform(df['Quality'])
df['Quality'].unique()


# In[10]:


df.head()


# In[11]:


df.drop(columns = ['A_id'],inplace = True)
df.head()


# In[13]:


df['Acidity'] = df['Acidity'].astype(float)
df.info()
import numpy as np
df = np.abs(df)


# In[14]:


X = df.drop(columns = ['Quality'])
Y = df['Quality']


# In[15]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.20)
print(f"x_train : {x_train.shape}\ny_train : {y_train.shape}\nx_test : {x_test.shape}\ny_test : {y_test.shape}")


# In[16]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train.values,y_train.values)


# In[17]:


model.score(x_train.values,y_train.values)


# In[18]:


from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier()
model1.fit(x_train,y_train)


# In[20]:


model1.score(x_train,y_train)


# In[21]:


model1.score(x_test,y_test)


# In[ ]:




