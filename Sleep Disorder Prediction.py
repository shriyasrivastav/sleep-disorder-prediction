#!/usr/bin/env python
# coding: utf-8

# # Sleep Disorder Prediction

# The aim of the project is to analyze the person's lifestyles and medical variables such as age, BMI, physical activity, sleep duration, blood pressure and many more, to predict the sleep disorder and its type.
# 
# ### About the Dataset
# The Sleep Health and Lifestyle Dataset comprises 400 rows and 13 columns, covering a wide range of variables related to sleep and daily habits. It includes details such as gender, age, occupation, sleep duration, quality of sleep, physical activity level, stress levels, BMI category, blood pressure, heart rate, daily steps, and the presence or absence of sleep disorders.
# 
# ### Key Features of the Dataset:
# - Comprehensive Sleep Metrics: Explore sleep duration, quality, and factors influencing sleep patterns.
# - Lifestyle Factors: Analyze physical activity levels, stress levels, and BMI categories.
# - Cardiovascular Health: Examine blood pressure and heart rate measurements.
# - Sleep Disorder Analysis: Identify the occurrence of sleep disorders such as Insomnia and Sleep Apnea.
# 
# ### Data Dictionary
# | Column Name | Description |
# | --- | --- |
# |Person_ID | Unique ID assigned to each person |
# |Gender|The gender of the person (Male/Female)|
# |Age | Age of the person in years |
# |Occupation | The occupation of the person |
# |Sleep_duration | The duration of sleep of the person in hours |
# |Quality_of_sleep | A subjective rating of the quality of sleep, ranging from 1 to 10|
# |Physical_activity | The level of physical activity of the person (Low/Medium/High) |
# |Stress Level| A subjective rating of the stress level, ranging from 1 to 10 |
# |BMI_category | The BMI category of the person (Underweight/Normal/Overweight/Obesity) |
# |Blood_pressure | The blood pressure of the person in mmHg |
# |Heart_rate | The heart rate of the person in beats per minute |
# |Daily Steps | The number of steps taken by the person per day |
# |Sleep_disorder | The presence or absence of a sleep disorder in the person (None, Insomnia, Sleep Apnea) |
# 
# 
# ### Details about Sleep Disorder Column:
# - None: The individual does not exhibit any specific sleep disorder.
# - Insomnia: The individual experiences difficulty falling asleep or staying asleep, leading to inadequate or poor-quality sleep.
# - Sleep Apnea: The individual suffers from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks.

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#loading the dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df.head()


# ## Data Preprocessing Part 1

# In[3]:


#checking for missing values
df.isnull().sum()


# In[4]:


#replacing the null values with 'None' in the column 'Sleep Disorder'
df['Sleep Disorder'].fillna('None', inplace=True)


# The nan/None value in sleep disorder stands for no sleep disorder, so it is not a missing value.

# In[5]:


#drop column Person ID
df.drop('Person ID', axis=1, inplace=True)


# In[6]:


#checking the number of unique values in each column
print("Unique values in each column are:")
for col in df.columns:
    print(col,df[col].nunique())


# #### Splitting the blood pressure into systolic and diastolic

# In[7]:


#spliting the blood pressure into two columns
df['systolic_bp'] = df['Blood Pressure'].apply(lambda x: x.split('/')[0])
df['diastolic_bp'] = df['Blood Pressure'].apply(lambda x: x.split('/')[1])
#droping the blood pressure column
df.drop('Blood Pressure', axis=1, inplace=True)


# In[8]:


#replacing normal weight with normal in BMI column
df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')


# In[9]:


df.head()


# ### Checking the unique values from each categorical column

# In[10]:


#unique values from categorical columns
print(df.Occupation.unique())
print('\n')
print(df['BMI Category'].unique())
print('\n')
print(df['Sleep Disorder'].unique())


# ## Explorative Data Analysis

# The EDA is divided into two phases:
# 
# Phase 1:  Understanding the data by plotting its variables
# 
# Phase 2: Understanding the correlation between the variables

# #### Phase 1

# In[11]:


fig,ax = plt.subplots(3,3,figsize=(20,10))
sns.countplot(x = 'Gender', data = df, ax = ax[0,0])
sns.histplot(x = 'Age', data = df, ax = ax[0,1], bins = 10)
sns.histplot(x = 'Sleep Duration', data = df, ax = ax[0,2], bins = 10)
sns.countplot(x = 'Quality of Sleep', data = df, ax = ax[1,0])
sns.histplot(x = 'Physical Activity Level', data = df, ax = ax[1,1], bins = 10)
sns.countplot(x = 'Stress Level', data = df, ax = ax[1,2])
sns.countplot(x = 'BMI Category', data = df, ax = ax[2,0])
sns.histplot(x = 'Daily Steps', data = df, ax = ax[2,1], bins = 10)
sns.countplot(x = 'Sleep Disorder', data = df, ax = ax[2,2])


# The number of males and females is almost equal, out of which majority of the people have age between 30-45 years. Most of the people have sleep quality greater than 5 which means there are getting sufficient sleep. Moreover, most of the people have normal BMI whci directly relates with the distribution of sleep disorder which shows equal number of people with and without sleep disorder.

# #### Phase 2

# #### Gender and Sleep Disorder

# In[12]:


#Gender count plot
sns.countplot(x = 'Gender', data = df, palette = 'hls', hue = 'Sleep Disorder').set_title('Gender and Sleep Disorder')


# Most of the males and females are not suffering from any sleep disorder. However females tend to have more sleep disorder as compared to males. The number of females suffering from Sleep Apnea is quite high as compared to males. But in contrast to that, greater number of males are suffering from Insomia as compared to females.

# ### Effect of Occupation on Sleep Disorder

# In[13]:


ax = sns.countplot(x = 'Occupation', data = df, hue = 'Sleep Disorder')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)


# From the graph it is clear that the occupation has huge impact on the sleep disorder. Nurses are more subjected to have Sleep Apenea as compared to other occupations and very few of them have no sleep disorder. After nurses, the next most affected occupation is the Salesperson, which counts for the  most suffering from Insomia followed by teachers. However there are some occupations where most of the people have very few instance of Sleep Apenea and Insomia such as Engineers, Doctors, Accountants, Lawyers. 
# The Software ENgineers and Managers are so less in number so I cannot say much about that, But the occupation Sales Representative has shown only Sleep Apenea and no Insomia or No sleep disorder. 

# ### BMI and Sleep Disorder

# In[14]:


sns.countplot(x = 'BMI Category', hue = 'Sleep Disorder', data = df, palette = 'Set1').set_title('BMI Category and Sleep Disorder')


# People with normal BMI are less likely to suffer from any sleep disorder. However, this is opposite in case of Overweight and Obese people. Overweight are more likely to suffer more from sleep disordera than Obese people.

# ## Data Preprocessing Part 2

# #### Label Encoding for categorical variables

# In[15]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[16]:


vars = ['Gender', 'Occupation','BMI Category','Sleep Disorder']
for i in vars:
    label_encoder.fit(df[i].unique())
    df[i] = label_encoder.transform(df[i])
    print(i,':' ,df[i].unique())


# ## Correlation Matrix Heatmap

# In[17]:


#Correlation Matrix Heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')


# ## Train Test Split

# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Sleep Disorder',axis=1), df['Sleep Disorder'], test_size=0.3, random_state=42)


# ## Model Building

# For predictiong the sleep disorder thriugh classification algorithms I will use the following algorithms:
# 1. Decision Tree Classifier
# 2. Random Forest Classifier

# ### Decision Tree Classifier

# In[19]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree


# Training the model with train dataset

# In[20]:


dtree.fit(X_train, y_train)


# In[21]:


#training accuracy
print("Training Accuracy:",dtree.score(X_train,y_train))


# ### Decision Tree Model Evalution

# In[22]:


d_pred = dtree.predict(X_test)
d_pred


# Using Confusion matrix heatmap to visualize the model accuracy

# In[23]:


from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_test, d_pred), annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


# The diagonal boxes show the count of true positive results, i.e correct predictions made by the model. The off-diagonal boxes show the count of false positive results, i.e incorrect predictions made by the model.

# ### Dsitribution plot for predicted and actual values

# In[24]:


ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(d_pred, hist=False, color="b", label="Fitted Values" , ax=ax)
plt.title('Actual vs Fitted Values for Sleep Disorder Prediction')
plt.xlabel('Sleep Disorder')
plt.ylabel('Proportion of People')
plt.show()


# The actual values are represented with red and the predicted ones with blue. As shown in the graph, the model's prediction are able to follow the curve of actual values but the predicted values are still different from actual ones. Therefore the model is not able to predict the values accurately.

# ##### Classification Report

# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_test, d_pred))


# The model gives pretty decent results with an accuracy of 87% and an average F1 score of 0.83. The model is able to predict the sleep disorder with a good accuracy.

# ### Random Forest Classifier

# In[26]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)


# Training the model with train dataset

# In[27]:


rfc.fit(X_train, y_train)


# In[28]:


#Training accuracy
print("Training accuracy: ",rfc.score(X_train,y_train))


# ### Random Forest Classifier Evaluation

# In[29]:


rfc_pred = rfc.predict(X_test)
rfc_pred


# Using confusion matrix heatmap to visualize the model accuracy

# In[30]:


#confusion matrix heatmap
sns.heatmap(confusion_matrix(y_test, rfc_pred), annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# The Random Forest Classifier model  has greater accuracy than the Decision Tree Classifier model. The diagonal boxes count for the True Positives i.e correct predictions, whereas the off-diagonal boxes show the count of false positive results, i.e incorrect predictions made by the model. Since the number of false positve value is less, it shows that the model is good at predicting the correct results.

# ### Distribution plot for predicted and acutal values

# In[31]:


ax = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(rfc_pred, hist=False, color="b", label="Predicted Values" , ax=ax)
plt.title('Actual vs Predicted values for Sleep Disorder')
plt.xlabel('Sleep Disorder')
plt.ylabel('Proportion of Patients')
plt.show()


# The Random forest classifier has improved accuracy as compared to the Decision Tree which is shown with the gap between the actual and predcited values which was wider incase of Descision Tree Classifier.

# #### Classification Report

# In[32]:


print(classification_report(y_test, rfc_pred))


# The Random Forest Classifier model has an accuracy of 89%  and an avergae F1 score of 0.86. From the metrics it is quite clear that the model is able to predict the sleep disorder quite effectively, with increased accuracy than Decision Tree Classifer.

# ## Conclusion
# 
# From the exploratory data analysis, I have concluded that the sleep orders depends upon three main factors that are gender, occupation and BMI of the patient. The males have more instance of Insomia whereas femlaes have more instances of Sleep Apnea. In addition the that people with occupation such as nurses are more prone to sleep disorders. The BMI of the patient also plays a vital role in the prediction of sleep disorders. The patients who are either Obese or overweight are more prone to sleep disorders.
# 
# Coming to the classfication models, both the models performed pretty good, however the Random Forest Classifier have excellent results with 89% accuracy.

# In[ ]:




