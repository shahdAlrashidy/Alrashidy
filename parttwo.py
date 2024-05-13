
# # Conclusions
# ### This dataset is very rich in information. Some limitations the dataset contains are null and zero values in some features. These zero and null values hinders the analysis and have to be removed or treated. For example null values is an obstacle which stopped me when I was analyzing the counts. Furthermore zero values creates false results during the correlation plots and computing the pearson correlation between Height, Weight and BMI. there are a number of student who smokes which is bad for a student in such a young age. Finally there is a positive correlation between some of the features of the midical student dataset but we also have negative ones.
# 
# ### After the Exploratort Data Analysis we can conclude that:
# 
# ####  1- Minimum heart rate = 60, average = 79.510192, and maximum = 99
# ####  2- Youngest student is 18 year old and the oldest is 34 year old  
# ####  3- The mean and the median in all columns are almost the same which leads to a normal distributions
# ####  4- Minimum BMI = 10.074837, average = 23.34013, and maximum = 44.355113
# ####  5- Students bodys' temperature is between 96 and 101 around 98 as a mean
# #### 6- most of the students are non-smokers
# #### 7- most of the students don't have diabetes
# #### 10- There is a strong positive relation between weight and BMI
# #### 11- the smoking habbit is more common in 28-year-old  students
# 

# # machine learning:

# In[29]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.model_selection import train_test_split 

df = pd.read_csv(r"Downloads\delaney_solubility_with_descriptors.csv")
y = df['logS']
x = df.drop('logS' , axis = 1)

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size =0.2, random_state = 100)
lr = LinearRegression()
lr.fit(x_train, y_train)
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
# y_train
# y_lr_train_pred

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2=r2_score(y_test,y_lr_test_pred)

# print(f"MLresultoflearn MSE (Train): ", lr_train_mse)
# print(f"LR MSE (Train): ", lr_train_r2)
# print(f"LR MSE (Train): ", lr_test_mse)
# print(f"LR MSE (Train): ", lr_test_r2)
lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse , lr_test_r2]).transpose()
# give Discreptive Names To Columns
lr_results.columns = ['Method','Training MSE' , 'Training R2' , 'Test MSE' ,'Test R2']
lr_results


# In[30]:


df


# In[31]:


import matplotlib.pyplot as plt
import numpy as np
plt.scatter(x=y_train , y = y_lr_train_pred, c='#7CAE00' ,alpha = 0.3)
plt.plot(y_train,y_train, '#F8766D')
# To draw line use numpy


# In[32]:


plt.style.available

