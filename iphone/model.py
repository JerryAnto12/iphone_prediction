#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd


# In[15]:


df1=pd.read_csv('raymon.csv')


# In[16]:


df1['Purchase Iphone'].replace(['iphone11','iphone12','iphone13','iphone13max'],[1,2,3,4],inplace=True)
df1['Gender'].replace(['Male','Female'],[0,1],inplace=True)


# In[17]:


df1.drop('Unnamed: 0',axis=1,inplace=True)


# In[18]:


X=df1.drop('Purchase Iphone',axis=1)
y=df1.pop('Purchase Iphone')
y=y.astype(int)


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[20]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
model=rf.fit(X_train, y_train)


# In[22]:


import pickle
pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


# In[ ]:





# In[ ]:




