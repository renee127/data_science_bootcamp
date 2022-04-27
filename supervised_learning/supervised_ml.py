#!/usr/bin/env python
# coding: utf-8

# Scenario:
# 
# Beta Bank customers are leaving and we've been asked to predict whether a customer will leave the bank soon using data on clients’ past behavior and termination of contracts with the bank.
# 
# <a class="anchor" id="table_of_contents">Overall Plan / Table of Contents</a>
# 
# - [Download and prepare the data](#prepare_data)
# - [Change data types](#datatypes)
# - [One hot encoding](#ohe)
# - [Divide dataframes into 3 groups](#split)
# - [Examine the balance of classes](#balance)
# - [Write print scores function](#scorefx)
# - [Train the model without balance adjustments](#unbalanced)
# - [Train different models and find the best one](#bestmodel)
# - [Build a model with the maximum possible F1 score (at least 0.59)](#f1)
# - [Perform final testing where F1 score >= 0.59](#testing)
# - [Measure the AUC ROC metric and compare it with F1](#aucroc)
# 
# Data source: https://www.kaggle.com/barelydedicated/bank-customer-churn-
# 
# Target: Exited — сustomer has left
# 
# Question: Predict if a customer will leave the bank

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle 
from sklearn.metrics import roc_curve
import random
import time


# import sys and insert code to ignore warnings 
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="prepare_data">Download and prepare the data</a>

# In[2]:


# load the data 
try:
    df = pd.read_csv('/datasets/Churn.csv')
except:
    print('ERROR: Unable to find or access file.')
df.head()


# In[3]:


# check columns, rows
df.shape


# In[4]:


# check for duplicates
df[df.duplicated()]


# In[5]:


# check info
df.info()


# In[6]:


# print composition of data types
df.dtypes.value_counts()


# In[7]:


# check for na values
df.isna().sum()


# In[8]:


# check general statistics for dataFrame
print('Statistics')
df.describe().T


# We note:
# - The file contains 10000 rows and 14 columns
# - There are no duplicated rows
# - Tenure column is missing 909 rows
# - Data types: 8 int64, 3 object, 3 float64 
# - Columns: 
#         - RowNumber — data string index
#         - CustomerId — unique customer identifier
#         - Surname — surname
#         - CreditScore — credit score
#         - Geography — country of residence
#         - Gender — gender
#         - Age — age
#         - Tenure — period of maturation for a customer’s fixed deposit (years)
#         - Balance — account balance
#         - NumOfProducts — number of banking products used by the customer
#         - HasCrCard — customer has a credit card
#         - IsActiveMember — customer’s activeness
#         - EstimatedSalary — estimated salary
# 
# Next steps:
# - Change column names to lower case
# - Assess if all columns are relavant to the analysis
# - Correct datatypes as needed
# - Investigate Tenure column's missing rows
# - Address columns (where is OHE appropriate?)
# - Address columns (where is label encoding appropriate?)

# In[9]:


# Map the lowering function to all column names
df.columns = map(str.lower, df.columns)
df.head(1)


# In[10]:


# verify customerid is unique for every user
print('Number of unique customerid values:', df.customerid.nunique())


# Are all columns relevant? rownumber simply indexes the customers, starting at one, so this may not be useful. customerid is also not useful, as every row corresponds to a unique customer already. surname could be useful if it identified familial relationships and family may influence each other to switch banks. But in a pool of 10000 individuals how can one identify relationships? We will investigate it further, but will likely drop surname.
# 
# creditscore, geography, gender, age, tenure, balance, numofproducts, hasccard, isactivemember, and estimatedsalary are all useful features. We will investigate surname to see if it may be helpful for analysis.

# In[11]:


# investigate surname column
print('Number of unique surnames', df.surname.nunique())
df.surname.value_counts()


# We will drop the rownumber, customerid, and surname columns since we will not be using them for analysis. Then we investigate the tenure column.

# In[12]:


# drop rownumber and surname from df
df = df.drop(['rownumber', 'customerid', 'surname'], axis=1)
df.head(1)


# In[13]:


# check info once again to check datatypes
df.dtypes


# In[14]:


# distribution of values for tenure column
df.tenure.value_counts(dropna=False)


# In[15]:


# explore missing values in tenure column
missing = df.tenure.isna().sum()
pct = missing/len(df)
print('Percent of total customers with a missing value in tenure = {:.2%}'.format(pct))
df[df.tenure.isna()].head()


# In[16]:


# investigate any correlations
df.corr()


# In[17]:


# examine the statistics of the tenure column
print('The mode for the tenure column:', df.tenure.mode())
df.tenure.describe().T


# We notice:
# - Less than 10% of the total customers are missing the value for tenure. Removing those rows is a possibility
# - There do not appear to be any correlations between tenure and other columns
# - The mode, most common value, is 1.0
# - The mean and median are quite close (4.99 and 5.0)
# 
# Since there are no obvious correlations and it is generally preferable to keep data instead of dropping it. 
# 
# We suspect the reason for the missing values is unknow. We will replace the missing values with the mean of tenure.

# In[18]:


# replace missing values in tenure with mean
#df.tenure = df.tenure.fillna(df.tenure.mean())
print('Number of missing values in tenure after replacement:',
     df.tenure.isna().sum()
     )
df.tenure.describe().T


# In[19]:


# will continue working on this in the future

# fill in the missing values in tenure with a randomly generated number

'''
limit = len(df)

# create a list of random numbers in range 0 to 10
#list = []
#for i in range(limit):
#    r=random.randint(0,10)
#    list.append(r)

# create a loop to fill NaN values with a number from the list
for i in range(limit):
    if np.isnan(df.tenure[i]) == True:  
        df['tenure'][i] = list[i]
    else:
         df.tenure[i] = df.tenure[i] 
df.tenure.value_counts()
'''


# We've replaced the missing values in tenure and now we will address the data types for the columns.
# 
# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="datatypes">**Change data types**</a>

# In[20]:


# refresh our memory on current data types
df.info()


# In[21]:


# change/downcast datatypes as appropriate
# For columns with low cardinality (the amount of unique values is lower than 50% of the count of these values)
# changing from object to category will help optimize memory and retrieval
# some of these could go to boolean instead, but will stick with integer as there doesn't seem a strong advantage

df.creditscore = pd.to_numeric(df.creditscore, downcast='integer')
df.geography = df.geography.astype('category')
df.gender = df.gender.astype('category')
df.age = pd.to_numeric(df.age, downcast='integer')
# since tenure is in years, we will change it to integer
df.tenure = df.tenure.apply(np.int)
df.tenure = df.tenure.astype('int16')
df.balance = pd.to_numeric(df.balance, downcast='float')
df.numofproducts = pd.to_numeric(df.numofproducts, downcast='integer')
df.hascrcard = pd.to_numeric(df.hascrcard, downcast='integer')
df.isactivemember  = pd.to_numeric(df.isactivemember, downcast='integer')
df.estimatedsalary = pd.to_numeric(df.estimatedsalary, downcast='float')
df.exited = pd.to_numeric(df.exited, downcast='integer')

# verify new data types
df.info()


# We've verified the new data types and note the reduced memory usage.
# 
# We note 2 columns with categorical information: geograpy and gender. We need a way to keep that information, but encode it.
# 
# One way is One Hot Encoding. It breaks the category into however many possible values and creates a new column for each possibility. One danger with OHE is the dummy trap, where the high correlation between all the columns can confuse the model. To avoid this, we drop the first column since its value may be inferred from the other columns. OHE isn't the best choice for tree-based models (decision trees, random forests) because the information gets lost as the algorithm travels deeper down the tree structure. 
# 
# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="ohe">**One hot encoding**</a>

# In[22]:


# use OHE to break out the catgeorical columns

print('Distribution of geography column before encoding')
print(df.geography.value_counts())

print('Distribution of gender column before encoding')
print(df.gender.value_counts())

df_ohe = pd.get_dummies(df, drop_first = True)

# verify new columns
df_ohe.head()


# In[23]:


# check counts of new columns
print('Germany')
print(df_ohe.geography_Germany.value_counts())
print('Spain')
# check counts of new columns
print(df_ohe.geography_Spain.value_counts())
print('Male')
# check counts of new columns
print(df_ohe.gender_Male.value_counts())


# We've verfied our newly created columns and the numbers from the original df match with the new columns. 
# 
# Next we will divide the dataframes into 3 groups (train, valid, test) for our models.
# 
# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="split">**Divide dataframes into 3 groups**</a>

# In[24]:


# split dataframes into train, valid, test data sets (3:1:1)

# divide df into target and features
target_ohe = df_ohe['exited']
features_ohe = df_ohe.drop('exited', axis=1)

# divide df_ohe (train, valid, test)
target_ohe_train, target_ohe_valid = train_test_split(target_ohe, test_size=0.2, random_state=12345) 
target_ohe_train, target_ohe_test = train_test_split(target_ohe_train, test_size=0.25, random_state=12345)
print('Verify sizes of newly divided target_ohe_')
print('train valid test')
print(len(target_ohe_train), len(target_ohe_valid), len(target_ohe_test))

features_ohe_train, features_ohe_valid = train_test_split(features_ohe, test_size=0.2, random_state=12345) 
features_ohe_train, features_ohe_test = train_test_split(features_ohe_train, test_size=0.25, random_state=12345)
print('\nVerify sizes of newly divided features_ohe_')
print('train valid test')
print(len(features_ohe_train), len(features_ohe_valid), len(features_ohe_test))


# We've verified the sizes of our dataframes are correct.
# 
# Now we briefly examine the balance of classes.
# 
# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="balance">**Examine the balance of classes**</a>

# In[25]:


# investigate balance of target in dataframes
print('Balance for df_ohe target (either 0 or 1)')
train_ohe_frequency = (target_ohe_train.value_counts(normalize=True) * 100).round(2)
valid_ohe_frequency = (target_ohe_valid.value_counts(normalize=True) * 100).round(2)
test_ohe_frequency = (target_ohe_test.value_counts(normalize=True) * 100).round(2)
print('% ohe Training Target balance of exited \n', train_ohe_frequency)
print('\n% ohe Validation Target balance of exited \n', valid_ohe_frequency)
print('\n% ohe Test Target balance of exited \n', test_ohe_frequency)

# https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html 


# Our sample is imbalanced, about 80% of our target column indicates no exit, so if we use this unbalanced data for our models our accuracy will be skewed towards no exit. Our model is predicting 0 in every instance because of the unbalanced data.
# 
# We will first create a print_scores function to increase efficiency and then test unbalanced models.
# 
# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="scorefx">**Write print scores function**</a>

# In[26]:


# create efficient print_scores function
def print_scores(true, predicted):
    print('Precision: ' + str((precision_score(true, predicted)* 100).round(2)) + '%')
    print('Recall: ' + str((recall_score(true, predicted)* 100).round(2)) + '%')
    print('F1 Score: ' + str((f1_score(true, predicted) * 100).round(2)) + '%')
    print('Accuracy Score: ' + str((accuracy_score(true, predicted) * 100).round(2)) + '%')
    print('Balanced Accuracy Score: ' + str((balanced_accuracy_score(true, predicted) * 100).round(2)) + '%')
    print('ROC AUC score: ' + str((roc_auc_score(true, predicted) * 100).round(2)) + '%')   


# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="unbalanced">**Train the model without balance adjustments**</a>

# In[27]:


# use dummy method to find a baseline
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(features_ohe_train, target_ohe_train)
# use print_scores fx to get scores
print_scores(target_ohe_valid, dummy_clf.predict(features_ohe_valid))


# In[28]:


# run a confusion matrix
confusion_matrix(target_ohe_valid, dummy_clf.predict(features_ohe_valid))


# As expected, our accuracy score is misleadingly high due to our unbalanced data. The F1 is actually zero, indicating a poorly performing model where either recall or precision or both are very low. In this case both precision and recall are 0.
# 
# The confusion matrix reveals many true positives, but almost 25% false positives. There are no true or false negatives. Our model is predicting 0 in every instance because of the unbalanced data.
# 
# Now we will begin building models that take the unbalanced data into account. We will begin with the simple linear model, and use a scaler to account for the different ranges (salary versus age for instance) of data. We don't want columns with larger numbers given more weight in the analysis.
# 
# **[Return to table of contents](#table_of_contents)**
# 
# 
# 
# <a class="anchor" id="bestmodel">**Train different models and find the best one**</a>
# 
# **Logistic Regression Linear Model**

# In[29]:


# scale model for linear using ohe data
scaler = StandardScaler()
scaler.fit(features_ohe_train)
features_ohe_train_scaled = scaler.transform(features_ohe_train)
features_ohe_valid_scaled = scaler.transform(features_ohe_valid)


# In[30]:


# create logistic regression model
lr = LogisticRegression(random_state=12345)
lr.fit(features_ohe_train_scaled, target_ohe_train)


# In[31]:


lr.predict(features_ohe_valid_scaled)


# In[32]:


print_scores(target_ohe_valid, lr.predict(features_ohe_valid_scaled))


# In[33]:


# run a confusion matrix
confusion_matrix(target_ohe_valid, lr.predict(features_ohe_valid_scaled))


# The model shows improvements over the baseline. Our precision, recall, and F1 score all increased. The accuracy stayed about the same, but the balanced accuracy score and ROC AUC scores improved. The confusion matrix reveals fewer false positives and finally some negtives. There are more true negatives than false negatives, indicating improvement.
# 
# We will attempt to adjust for the unbalanced data by adding class_weight='balanced' to the linear regression initialization. 

# In[34]:


# create logistic regression model using class_weight='balanced'
lr = LogisticRegression(random_state=12345, class_weight='balanced')
lr.fit(features_ohe_train_scaled, target_ohe_train)


# In[35]:


lr.predict(features_ohe_valid_scaled)
print_scores(target_ohe_valid, lr.predict(features_ohe_valid_scaled))
# run a confusion matrix
confusion_matrix(target_ohe_valid, lr.predict(features_ohe_valid_scaled))


# We observe a great increase in recall, F1 score, and the ROC AUC score. The accuracy score dropped, as did the precision. The confusion matrix shows an overall better distribution. However, the number of false negatives rose precipitiously. 
# 
# Next we will explore the decision tree classifier.
# 
# **Decision Tree Classifier**

# In[36]:


# build decision tree model
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_ohe_train, target_ohe_train)


# In[37]:


# display scores
print_scores(target_ohe_valid, model.predict(features_ohe_valid))


# In[38]:


# run a confusion matrix
confusion_matrix(target_ohe_valid, model.predict(features_ohe_valid))


# We note even without balancing of classes, the F1 score is above 50%. The false positive and false negatives are still proportionally high. We will add class_weight=balanced to explore how this changes.

# In[39]:


# build decision tree model
model = DecisionTreeClassifier(random_state=12345, class_weight='balanced')
model.fit(features_ohe_train, target_ohe_train)
# display scores
print_scores(target_ohe_valid, model.predict(features_ohe_valid))
# run a confusion matrix
confusion_matrix(target_ohe_valid, model.predict(features_ohe_valid))


# Initializing class wight to balanced didn't change the metrics in any appreciable way. Next we will try random forest. It should have higher accuracy.
# 
# **Random Forest**

# In[40]:


# build random forest model for encode data
model = RandomForestClassifier(random_state=12345)
model.fit(features_ohe_train, target_ohe_train)


# In[41]:


# display scores
print_scores(target_ohe_valid, model.predict(features_ohe_valid))
# run a confusion matrix
confusion_matrix(target_ohe_valid, model.predict(features_ohe_valid))


# The F1 score is higher than in the other two models. Now we will add class_weight=balanced.

# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="f1">**Build a model with the maximum possible F1 score (at least 0.59)**</a>

# In[42]:


# build random forest model 
model = RandomForestClassifier(random_state=12345, class_weight='balanced')
model.fit(features_ohe_train, target_ohe_train)
# display scores
print_scores(target_ohe_valid, model.predict(features_ohe_valid))
# run a confusion matrix
confusion_matrix(target_ohe_valid, model.predict(features_ohe_valid))


# The F1 score actually went down. Next we will explore the optimal depth and n-estimators.

# In[43]:


print('Accuracy for random forest classifier model\n')
print('What is the best max_depth?')
# print('max_depth                    F1 score')

# set up list for accuracy score
accuracy_list = []

# find the accuracy score when n_estimators is between 1 and 100
for n in range(1, 100):
        # notice need random_state=12345 here
        model = RandomForestClassifier(random_state=12345, class_weight ='balanced',
                                       max_depth = n) 

        # train the model/fit model 
        model.fit(features_ohe_train, target_ohe_train)
        
        # find the predictions using validation set 
        # notice not using score...
        predictions_valid = model.predict(features_ohe_valid)
        
        # calculate F1 score
        acc_score = f1_score(target_ohe_valid, predictions_valid)          
        
        # print n value and accuracy score
        # print("max_depth =", n, ": ", acc_score)        
        
        # add n value and accuracy score to list
        accuracy_list.append(acc_score)
        
# find the max n_estimator and save it as best_n_estimator
max_accuracy = max(accuracy_list)

# add one to calculation because index begins at 0
best_max_depth = accuracy_list.index(max_accuracy) + 1

# print n_estimator and accuracy score
print("The best performing depth =", best_max_depth, ": ", max_accuracy)
print('')


# In[44]:


print('Accuracy for random forest classifier model\n')
print('What is the best n_estimators?')
#print('n_estimators           F1 score')

# set up list for accuracy score
accuracy_list = []

# find the accuracy score when n_estimators is between 50 and 250
for n in range(1, 20):
        # notice need random_state=12345 here
        model = RandomForestClassifier(random_state=12345, class_weight ='balanced',
                                       n_estimators = n) 

        # train the model/fit model 
        model.fit(features_ohe_train, target_ohe_train)
        
        # find the predictions using validation set 
        # notice not using score...
        predictions_valid = model.predict(features_ohe_valid)
        
        # calculate F1 score
        acc_score = f1_score(target_ohe_valid, predictions_valid)      
        
        # calculate accuracy score
        #acc_score = accuracy_score(target_encode_valid, predictions_valid)      
        
        # print n value and accuracy score
        # print("n_estimators =", n, ": ", acc_score)        
        
        # add n value and accuracy score to list
        accuracy_list.append(acc_score)
        
# find the max n_estimator and save it as best_n_estimator
max_accuracy = max(accuracy_list)

# add one to calculation because index begins at 0
best_n_estimator = accuracy_list.index(max_accuracy) + 1

# print n_estimator and accuracy score
print("The best performing n_estimators =", best_n_estimator, ": ", max_accuracy)
print('')


# In[45]:


# build random forest model using best max_depth, n_estimators
model = RandomForestClassifier(random_state=12345, class_weight='balanced',
                              max_depth = 10, n_estimators = 17)
model.fit(features_ohe_train, target_ohe_train)
# display scores
print_scores(target_ohe_valid, model.predict(features_ohe_valid))
# run a confusion matrix
confusion_matrix(target_ohe_valid, model.predict(features_ohe_valid))


# Here we have the highest F1 score yet, well above the 59% threshold. Finally, let's unsample the target data and see if we can improve the F1 score.

# In[46]:


# upsample target data

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target ==1]

    return features_upsampled, target_upsampled

features_zeros = features_ohe_train[target_ohe_train == 0]
features_ones = features_ohe_train[target_ohe_train == 1]
target_zeros = target_ohe_train[target_ohe_train == 0]
target_ones = target_ohe_train[target_ohe_train == 1]

repeat = 10
features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)

features_upsampled, target_upsampled = upsample(
    features_ohe_train, target_ohe_train, 10
)

# print(features_upsampled.shape)
# print(target_upsampled.shape)

# shuffle the unsampled
features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345) 

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(
    features_ohe_train, target_ohe_train, 10
)

model = RandomForestClassifier(random_state=12345, class_weight='balanced',
                              max_depth = 10, n_estimators = 17)
model.fit(features_upsampled, target_upsampled)

predicted_valid = model.predict(features_ohe_valid)

# display scores
print_scores(target_ohe_valid, model.predict(features_ohe_valid))
# run a confusion matrix
confusion_matrix(target_ohe_valid, model.predict(features_ohe_valid))


# Upsampling actually decreased our F1 score. We will continue go back to using the previous model.
# 
# Finally we have a model we can test. The F1 score is well above the 59% threshold, the accuracy score is above the baseline we tested initially, and the confusion matrix demonstrates a greater number of true positives/negatives than the negative postives/negatives. 
# 
# We will use our test data to verify.

# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="testing">**Perform final testing where F1 score >= 0.59**</a>

# In[47]:


# build random forest model using best max_depth, n_estimators
model = RandomForestClassifier(random_state=12345, class_weight='balanced',
                              max_depth = 10, n_estimators = 17)
model.fit(features_ohe_train, target_ohe_train)
# display scores
print_scores(target_ohe_valid, model.predict(features_ohe_valid))
# run a confusion matrix
confusion_matrix(target_ohe_valid, model.predict(features_ohe_valid))


# In[48]:


# display scores
print_scores(target_ohe_test, model.predict(features_ohe_test))
# run a confusion matrix
confusion_matrix(target_ohe_test, model.predict(features_ohe_test))


# We find the F1 score is over 59% for the test data, indicating the high correlation between precision and recall. The accuracy score is above the baseline, not by much, but some. The ROC AUC score is pretty robust. A ROC of 50% indicates a model created by chance. 
# 
# **[Return to table of contents](#table_of_contents)**
# 
# <a class="anchor" id="aucroc">Measure the AUC ROC metric and compare it with F1</a>

# In[49]:


probabilities_valid = model.predict_proba(features_ohe_valid)
probabilities_one_valid = probabilities_valid[:, 1]

# notice used probabilities_one_valid
fpr, tpr, thresholds = roc_curve(target_ohe_valid, probabilities_one_valid) 

plt.figure()
plt.plot(fpr, tpr)
# < plot the graph >

# ROC curve for random model (looks like a straight line)
plt.plot([0, 1], [0, 1], linestyle='--')

# < use the functions plt.xlim() and plt.ylim() to
#   set the boundary for the axes from 0 to 1 >
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

# use the functions plt.xlabel() and plt.ylabel() to
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# add the heading "ROC curve" with the function plt.title() 
plt.title('ROC curve')
plt.show()


# The ROC curve indicates our model will consistently predict those customers who will stay with the bank and those that likely to move on. While not perfect, our model is far better than chance. We recommend using the random forest classification with weighted balance in this situation.

# References
# 
# - [Downcasting data types and memory](https://hackersandslackers.com/downcast-numerical-columns-python-pandas/)
