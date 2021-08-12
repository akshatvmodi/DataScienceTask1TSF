#!/usr/bin/env python
# coding: utf-8

# GRIP - The Sparks Foundation

# Data Science and Business Analytics Intern

# AUTHOR:- AKSHAT VIPUL MODI

# Task1: Prediction using Supervised ML

# - Predict the percentage of an student based on the no. of study hours. - This is a simple linear regression task as it involves just 2 variables. - You can use R, Python, SAS Enterprise Miner or any other tool. - Data can be found at http://bit.ly/w-data

# In[37]:


# Predict the percentage of an student based on the no. of study hours


# In[38]:


# importing all libraries required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for plotting different graphs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


student_data = pd.read_csv("http://bit.ly/w-data") # loading data from the link given by the TSF


# In[4]:


student_data.shape # checking the records present in the dataset


# In[40]:


student_data.info() # gives the details of the dataset


# In[41]:


student_data.head() # will give first five records


# In[42]:


student_data.describe() # will give the different statistical values


# In[43]:


student_data.corr() # correlation between two variables


# In[44]:


student_data.plot(kind='scatter',x='Hours',y='Scores')
plt.title("Hours vs Scores")    # plotting the scatter plot for the dataset
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[45]:


# Preparing the Data


# In[46]:


X = student_data.iloc[:, :-1].values  
Y = student_data.iloc[:, 1].values  


# In[47]:


#Splitting the data set as Training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.2, random_state=0)


# In[48]:


#Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)


# Predicting the value of test set results

# In[49]:


#Predicting the values of test set
Score_prediction = model.predict(X_train)


# In[50]:


print(model.coef_) # printing the co-efficient


# In[51]:


print(model.intercept_) # intercept of the model


# In[52]:


# PLotting the training set
plt.scatter(X_train, Y_train, color= 'blue')
plt.plot(X_train, model.predict(X_train), color = 'red')
plt.title('Hours vs Scores(Training Data Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[53]:


#Predicting the value of score using hour that is alredy in the data set
model.predict([[5.1]])


# In[54]:


Y_test


# In[55]:


Y_predict=model.predict(X_test)

Y_predict


# In[56]:


dfrm= pd.DataFrame(Y_test,Y_predict)

df1 = pd.DataFrame({'Actual':Y_test, 'Predicted':Y_predict})
df1


# What will be predicted score if a student studies for 9.25 hrs/ day?

# In[57]:


#Predicting through given data
Hrs = 9.25
predct = model.predict([[Hrs]])
print("No of Hours = {}".format(Hrs))
print("Predicted Score = {}".format(predct[0]))


# In[58]:


#Checking random test data
Hrs1 = 8
predct1 = model.predict([[Hrs1]])
print("No of Hours = {}".format(Hrs1))
print("Predicted Score = {}".format(predct1[0]))


# In[59]:


#Finding mean absolute error


# In[60]:


import sklearn.metrics as metrics 
from sklearn.metrics import confusion_matrix, accuracy_score
import math
print("Mean Squared Error:",metrics.mean_squared_error(Y_test,Y_predict))
print("Mean Absolute error :",metrics.mean_absolute_error(Y_test,Y_predict))#It will give the mean errorof  the model
print("Root Mean Squared Error:",math.sqrt(metrics.mean_squared_error(Y_test,Y_predict)))


# In[61]:


model.score(X_test,Y_test) # checking the score of the model


# THANK YOU!

# In[ ]:




