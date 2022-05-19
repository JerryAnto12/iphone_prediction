#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("iphone_purchase_list.csv")


# In[2]:


df['Purchase Iphone'].value_counts()


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df['Salary'].value_counts()


# In[6]:


df.isnull().sum()
    


# In[7]:


df['Salary'].value_counts()


# In[8]:


df['Purchase Iphone']


# In[9]:


df['Purchase Iphone'].value_counts()


# In[10]:


iphone11=df.loc[(df['Purchase Iphone']==1) & (df['Salary']<50000)]
iphone11=pd.DataFrame(iphone11)
iphone11['Purchase Iphone']='iphone11'
iphone11.shape


# In[11]:


iphone12=df.loc[(df['Purchase Iphone']==1)&(df['Salary']>50000) & (df['Salary']<80000)]
iphone12=pd.DataFrame(iphone12)
iphone12['Purchase Iphone']='iphone12'
iphone12.shape


# In[12]:


iphone13=df.loc[(df['Purchase Iphone']==1)&(df['Salary']>80000) & (df['Salary']<100000)]
iphone13=pd.DataFrame(iphone13)
iphone13['Purchase Iphone']='iphone13'
iphone13.shape


# In[13]:


iphone13max=df.loc[(df['Purchase Iphone']==1)&(df['Salary']>100000)]
iphone13max=pd.DataFrame(iphone13max)
iphone13max['Purchase Iphone']='iphone13max'
iphone13max.shape


# In[14]:


noiphone=df.loc[(df['Purchase Iphone']==0)]
noiphone.shape


# In[15]:


iphone=pd.concat([iphone11,iphone12,iphone13,iphone13max,noiphone],axis=0)


# In[16]:


iphone


# In[17]:


iphone.to_csv('raymon.csv')


# In[18]:


df1=pd.read_csv('raymon.csv')
df1


# In[19]:


df1['Purchase Iphone'].replace(['iphone11','iphone12','iphone13','iphone13max'],[1,2,3,4],inplace=True)
df1['Gender'].replace(['Male','Female'],[0,1],inplace=True)


# In[20]:


df1.head()


# In[21]:


df1.drop('Unnamed: 0',axis=1,inplace=True)


# In[22]:


df1


# In[23]:


df1['Purchase Iphone'].value_counts()


# In[24]:


df['Gender'].value_counts()


# In[25]:


X=df1.drop('Purchase Iphone',axis=1)
y=df1.pop('Purchase Iphone')


# In[26]:


y=y.astype(int)


# ### Splitting data

# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# ### MultinomialNB

# In[28]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()


# In[29]:


clf.fit(X_train,y_train)


# In[30]:


y_pred = clf.predict(X_test)


# In[31]:


from sklearn.metrics import accuracy_score, classification_report
print("Accuracy Score is =", accuracy_score(y_test, y_pred))


# ### SVC

# In[32]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import warnings


# In[33]:


model = OneVsRestClassifier(SVC())
   
# Fitting the model with training data
model.fit(X_train, y_train)
   
# Making a prediction on the test set
prediction = model.predict(X_test)
   
# Evaluating the model
print(f"Test Set Accuracy : {accuracy_score(y_test, prediction) * 100} %\n\n")
print(f"Classification Report : \n\n{classification_report( y_test, prediction)}")


# ### Logistic One VS Rest

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


# In[35]:


scaler = StandardScaler()
X_trainss = scaler.fit_transform(X_train)
X_testss=scaler.transform(X_test)


# In[36]:


lo = LogisticRegression(random_state=0, multi_class='ovr')


# In[37]:


model = lo.fit(X_trainss, y_train)
y_predss=model.predict(X_testss)


# In[38]:


model.predict_proba(X_testss)


# In[39]:


laccuracy=accuracy_score(y_test, y_predss) 


# In[40]:


print(f"Test Set Accuracy : {accuracy_score(y_test, y_predss) * 100} %\n\n")
print(f"Classification Report : \n\n{classification_report( y_test,y_predss)}")


# In[ ]:





# ### Decision Tree

# In[41]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)


# In[42]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, dtree_predictions)
cm


# In[43]:


dtaccuracy = dtree_model.score(X_test, y_test)
dtaccuracy


# ### Support Vector Machine

# In[44]:


from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
 
# model accuracy for X_test 
svmaccuracy = accuracy_score(y_test,svm_predictions)

# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)


# In[45]:


svmaccuracy


# ### KNeighbourClassifier

# In[46]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
knn_predictions = knn.predict(X_test) 
# accuracy on X_test
knnaccuracy = accuracy_score(y_test,knn_predictions)
print (knnaccuracy)
 
# creating a confusion matrix

cm = confusion_matrix(y_test, knn_predictions)
cm


# ### Naivebias

# In[47]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
 
# accuracy on X_test
gnbaccuracy = accuracy_score(y_test,gnb_predictions )
print(gnbaccuracy)
 
# creating a confusion matrix
cm = confusion_matrix(y_test, gnb_predictions)
cm

gnb.predict([[0,54,1]])


# In[48]:


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB().fit(X_train, y_train)
y_predmnb = mnb.predict(X_test)
 
# accuracy on X_test
mnbaccuracy = accuracy_score(y_test, y_predmnb)
print(mnbaccuracy)


# ### Random Forest

# In[49]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
rf.fit(X_train, y_train)
#Predicting the Test set results
y_pred = rf.predict(X_test)
rfaccuracy =accuracy_score(y_test, y_pred)


# In[50]:


rfaccuracy


# ### Feature_importances

# In[51]:


import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini
rfc.fit(X_train, np.ravel(y_train,order='C'))
fi=rfc.feature_importances_
for i,v in enumerate(fi):
    print("feature: %od, Score: %.5f"%(i,v))
rfcpred = rfc.predict(X_test)
cnf_matrix = confusion_matrix(y_test, rfcpred)
print(cnf_matrix)
print("Accuracy:",accuracy_score(y_test, rfcpred))


# In[52]:


from sklearn.naive_bayes import MultinomialNB



clf =  MultinomialNB().fit(X_train, y_train)
clf_predictions = clf.predict(X_test)
 
# accuracy on X_test
mnbaccuracy = accuracy_score(y_test,clf_predictions)
print(mnbaccuracy)
 


# In[ ]:





# ### ACCURACIES OF VARIOUS MODELS

# In[53]:


print(f" Accuracy of KNN model : {knnaccuracy * 100} %\n\n")
print(f" Accuracy of SVM model : {svmaccuracy * 100} %\n\n")
print(f" Accuracy of Logistic Regression model : {laccuracy * 100} %\n\n")
print(f" Accuracy of Decision Tree model : {dtaccuracy * 100} %\n\n")
print(f" Accuracy of Naivebias GaussianNB model : {gnbaccuracy * 100} %\n\n")
print(f" Accuracy of Naivebias MultinomialNB model : {mnbaccuracy * 100} %\n\n")
print(f" Accuracy of Random Forest : {rfaccuracy * 100} %\n\n")


# In[54]:


rf.predict([[0,34,70000]])


# In[ ]:





# ## Visualizations

# In[55]:


import matplotlib.pyplot as plt


# In[56]:


x = df1['Salary']
y = df1['Age']

plt.scatter(x,y, marker="o", s=25, edgecolor="k")
plt.show()


# In[57]:


import seaborn as sns


# In[58]:


sns.relplot(x='Age',y='Salary',hue='Purchase Iphone',data=df)


# In[59]:


y=df1['Salary']
sns.catplot(x='Gender',y='Salary',hue='Purchase Iphone',kind='bar',data=df)


# In[61]:


sns.countplot(x='Age',data=df1,palette='mako_r')

plt.show


# In[62]:


old=len(df1[df1.Age>=30])
teen=len(df1[df1.Age<=29])
print('Percentage of old customers bought iphone(age>30):{:.2f}%'.format((old/(len(df1.Age))*100)))
print('Percentage of young customers bought iphone(age<30):{:.2f}%'.format((teen/(len(df1.Age))*100)))


# In[63]:


less_sal=len(df1[df1.Salary>=30000])
high_sal=len(df1[df1.Salary<=30000])
print('Percentage of customers with high salary(more than 30k):{:.2f}%'.format((less_sal/(len(df1.Salary))*100)))
print('Percentage of customers with not so high salary(not more than 30k) :{:.2f}%'.format((high_sal/(len(df1.Salary))*100)))


# In[ ]:




