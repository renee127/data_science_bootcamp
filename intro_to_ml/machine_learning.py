#!/usr/bin/env python
# coding: utf-8

# Scenario:
# 
# Mobile carrier Megaline has 2 newer plans, Smart or Ultra, but many of their subscribers still use a legacy plan.
# 
# As an analysts for Megaline, we've been asked to create a machine learning model that recommends an appropriate plan based on data about the behavior of those subscribers who've already switched. 
# 
# Accuracy counts. Our model needs an **accuracy >= 75%**.
# 
# This is a classification task because our **target (is_ultra)** is categorical: Ultra - 1, Smart - 0
# 
# Our plan:
# - download the data
# - investigate the data (it should already be preprocessed)
# - split the data into train, validation, and test data sets
# - create models / test different hyperparameters
# - check the accuracy using the test data set
# - sanity check the model
# - discuss findings
# 
# Because this is a business classification task where accuracy is most important, we will start with the Random Forest Classifier and test other models if needed.
# 
# Our question becomes: Can we predict which plan to recommend based on behavior of users who've switched to one of the new plans?

# In[1]:


# import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier


# In[2]:


# import sys and insert code to ignore warnings 
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[3]:


# load the data 
try:
    df = pd.read_csv('/datasets/users_behavior.csv')
except:
    print('ERROR: Unable to find or access file.')
df.head()


# In[4]:


# check info
df.info()


# In[5]:


# check for na values
df.isna().sum()


# In[6]:


# check for duplicates
df[df.duplicated()]


# In[7]:


df.shape


# Data description: No missing values, duplicate rows, or other issues noted across the 5 columns and 3214 rows.
# 
# - сalls — number of calls
# - minutes — total call duration in minutes
# - messages — number of text messages
# - mb_used — Internet traffic used in MB
# - is_ultra — plan for the current month (Ultra - 1, Smart - 0)

# In[8]:


# split data into train, valid, test data sets (3:1:1)
# first split train test into df_train, df_valid, then divide into df_train, df_test
df_train, df_valid = train_test_split(df, test_size=0.2, random_state=12345) 
# print(len(df), len(df_train), len(df_valid))
df_train, df_test = train_test_split(df_train, test_size=0.25, random_state=12345)
print('Verify sizes of newly divided dataframes\n')
print('train valid test\n')
print(len(df_train), len(df_valid), len(df_test))
print('\nCalculate means of is_ultra in each data set')
print('train              valid                  test\n')
print(df_train.is_ultra.mean(), df_valid.is_ultra.mean(), df_test.is_ultra.mean())


# Our original data frame is divided into 3 new data frames with a ration of train(3):valid(1):test(1). In other words, 60% of the sample is in the train data set, 20% in the valid and 20% in the test.
# 
# We also note that in each data set around 30% of the populations have the Ultra plan. This distribution verifies that the df dataset has been divided appropriately, at least as far as is_ultra is concerned.

# In[9]:


# create features dfs where is_ultra, the target is dropped
# create target dfs with only is_ultra
print('Verify rows and columns of train and valid sets\n')

features_train = df_train.drop(['is_ultra'], axis=1)
target_train = df_train['is_ultra']
print('features_train', features_train.shape)
print('target_train', target_train.shape)

features_valid = df_valid.drop(['is_ultra'], axis=1)
target_valid = df_valid['is_ultra']
print('features_valid', features_valid.shape)
print('target_valid', target_valid.shape)

features_test = df_test.drop(['is_ultra'], axis=1)
target_test = df_test['is_ultra']
print('features_test', features_test.shape)
print('target_test', target_test.shape)


# In[10]:


# create random forest classifier model

# create loop for n_estimators

print('Accuracy for random forest classifier model\n')
print('n_estimators           accuracy')

# set up list for accuracy score
accuracy_list = []

# find the accuracy score when n_estimators is between 1 and 100
for n in range(1, 101):
        # notice need random_state=12345 here
        model = RandomForestClassifier(random_state=12345, n_estimators = n) 

        # train the model/fit model 
        model.fit(features_train, target_train)
        
        # find the predictions using validation set 
        # notice not using score...
        predictions_valid = model.predict(features_valid)
        
        # calculate accuracy score
        acc_score = accuracy_score(target_valid, predictions_valid)      
        
        # print n value and accuracy score
        print("n_estimators =", n, ": ", acc_score)        
        
        # add n value and accuracy score to list
        accuracy_list.append(acc_score)


# In[11]:


# find the max n_estimator and save it as best_n_estimator
max_accuracy = max(accuracy_list)

# add one to calculation because index begins at 0
best_n_estimator = accuracy_list.index(max_accuracy) + 1

# print n_estimator and accuracy score
print("The best performing n_estimators =", best_n_estimator, ": ", max_accuracy)
print('')

print('Our first choice to make this model is the random forest classifier because '
      'of the high accuracy. We create a loop to run through n_estimators between 1 and 100. '
      'We note the accuracy score is generally 78% to 79%. \nThe best result occurs when the '
      'n-estimators =', best_n_estimator, 'with an accuracy of: {:.2%}'.format(max_accuracy))
print('We will use this n_estimators for a final test.')


# In[12]:


# test random forest classifier model using best result
# and compare with train data set, test data set

# notice need random_state=12345 here
model = RandomForestClassifier(random_state=12345, n_estimators = best_n_estimator) 

# train the model/fit model 
model.fit(features_train, target_train)

# find the predictions using validation set 
predictions_valid = model.predict(features_valid)

valid_accuracy = accuracy_score(target_valid, predictions_valid)

predictions_train = model.predict(features_train)
predictions_test = model.predict(features_test)

# write code for training set calculations here 
accuracy = accuracy_score(target_train, predictions_train)

# write code for test set calculations here
test_accuracy = accuracy_score(target_test, predictions_test)

print('Accuracy\n')
print('Validation set: {:.2%}'.format(valid_accuracy))
print('Training set: {:.2%}'.format(accuracy))
print('Test set: {:.2%}'.format(test_accuracy))


# As we expect, the model scores almost 100% on the training set. Both the validation set and the test set are over 75%, our threshold, so this may be a good choice for a model to use. 
# 
# However, we would also like to examine the decision tree classifier model (generally known for lower accuracy but greater speed) and the logistic regression model (known for medium accuracy). 

# In[13]:


# create decision tree classifier model

# create loop for max_depth

print('Accuracy for decision tree classifier model\n')
print('max_depth      accuracy')

# set up list for accuracy score
accuracy_list = []

for depth in range(1, 21):
        # create a model, specify max_depth=depth 
        # notice need random_state=12345 here
        model = DecisionTreeClassifier(random_state=12345, max_depth = depth)

        # train the model/fit model 
        model.fit(features_train, target_train)
        
        # find the predictions using validation set
        # notice not using score...
        predictions_valid = model.predict(features_valid)
        
        # calculate accuracy score
        acc_score = accuracy_score(target_valid, predictions_valid)      
        
        # print n value and accuracy score
        print("max_depth =", depth, ": ", acc_score)        
        
        # add n value and accuracy score to list
        accuracy_list.append(acc_score)


# In[14]:


# find the max depth and save it as best_max_depth
max_accuracy = max(accuracy_list)

# add one to calculation because index begins at 0
best_max_depth = accuracy_list.index(max_accuracy) + 1

# print best max depth and accuracy score
print("The best performing max_depth =", best_max_depth, ": ", max_accuracy)  

print('We create a loop to run through max_depths between 1 and 20 for the decision tree classifier. '
      'We note the accuracy score peaks around 78%. \nThe best result occurs when the '
      'n-estimators =', best_max_depth, 'with an accuracy of: {:.2%}'.format(max_accuracy))
print('We will use this best_max_depth for a final test.')


# In[15]:


# test decision tree classifier model using best result of max_depth = 7
# and compare with train data set, test data set

# notice need random_state=12345 here
model = DecisionTreeClassifier(random_state=12345, max_depth = best_max_depth) 

# train the model/fit model 
model.fit(features_train, target_train)

# find the predictions using validation set 
predictions_valid = model.predict(features_valid)

valid_accuracy = accuracy_score(target_valid, predictions_valid)

predictions_train = model.predict(features_train)
predictions_test = model.predict(features_test)

# write code for training set calculations here 
accuracy = accuracy_score(target_train, predictions_train)

# write code for test set calculations here
test_accuracy = accuracy_score(target_test, predictions_test)

print('Accuracy\n')
print('Validation set: {:.2%}'.format(valid_accuracy))
print('Training set: {:.2%}'.format(accuracy))
print('Test set: {:.2%}'.format(test_accuracy))


# Once again we note the highest accuracy is for the training set, but it is far less than the 99% of the random forest classifier. Even though the validation and test sets are over 75%, we still believe the best model is the random forest classifier. Finally, we will check out the logistic regression model.

# In[16]:


# create logistic regression model

model = LogisticRegression(random_state=12345, solver='liblinear')
# train the model/fit model 
model.fit(features_train, target_train)

# find the predictions using validation set 
# notice not using score...
predictions_valid = model.predict(features_valid)

# train the model/fit model 
model.fit(features_train, target_train)

# find the predictions using validation set 
predictions_valid = model.predict(features_valid)

valid_accuracy = accuracy_score(target_valid, predictions_valid)

predictions_train = model.predict(features_train)
predictions_test = model.predict(features_test)

# write code for training set calculations here 
accuracy = accuracy_score(target_train, predictions_train)

# write code for test set calculations here
test_accuracy = accuracy_score(target_test, predictions_test)

print('Accuracy\n')
print('Validation set: {:.2%}'.format(valid_accuracy))
print('Training set: {:.2%}'.format(accuracy))
print('Test set: {:.2%}'.format(test_accuracy))


# The results of the logistic regression model are disappointing and don't even reach our 75% threshold.
# 
# We recommend the RandomForestClassifier model using the best performing n_estimators value. 
# 
# We will perform a sanity check on the selected test data below: 

# In[17]:


# sanity check the test data

# we are using the test data, divided and filtered as below:
# features_test = df_test.drop(['is_ultra'], axis=1)
# target_test = df_test['is_ultra']

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(features_test, target_test)
dummy_clf.predict(features_test)
dummy_clf.score(features_test, target_test)

sanity_score = dummy_clf.score(features_test, target_test)
print('Sanity check of test data: {:.2%}'.format(sanity_score))


# In[18]:


print('The RandomForestClassifier (random_state=12345, n_estimators =', best_n_estimator,') '
      'reliably (over 75% of the time) predicts which plan to recommend based on the behavior '
      'of users who\'ve switched to one of the new plans. \n\nOur selected model passes '
      'the sanity check when we use the dummy classifier to determine the percent correct '
      'by chance alone for this classification/catagorical problem.'
      '\n\nOur score, {:.2%}'.format(max_accuracy), 'is greater than the ' 
      'sanity score {:.2%}'.format(sanity_score))


# Refrences
# 
# [Ways to divide a data set in 3 proportions](https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test)
# 
# <a href="https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html"> DummyClassifier </a>
# 
# 
# 

# In[19]:


# alternative way to divide
# train, valid, test = \
#              np.split(df.sample(frac=1, random_state=12345), 
#                       [int(.6*len(df)), int(.8*len(df))])
# print(len(train), len(valid), len(test))
# results 1928 643 643

