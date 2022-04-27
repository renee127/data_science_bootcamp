#!/usr/bin/env python
# coding: utf-8

# Rusty Bargain used car sales service is developing an app to attract new customers. In that app, you can quickly find out the market value of your car. You have access to historical data: technical specifications, trim versions, and prices. You need to build the model to determine the value. 
# 
# Rusty Bargain is interested in:
# 
# - the quality of the prediction;
# - the speed of the prediction;
# - the time required for training
# 
# ---
# Features
# 
# - DateCrawled — date profile was downloaded from the database
# - VehicleType — vehicle body type
# - RegistrationYear — vehicle registration year
# - Gearbox — gearbox type
# - Power — power (hp)
# - Model — vehicle model
# - Mileage — mileage (measured in km due to dataset's regional specifics)
# - RegistrationMonth — vehicle registration month
# - FuelType — fuel type
# - Brand — vehicle brand
# - NotRepaired — vehicle repaired or not
# - DateCreated — date of profile creation
# - NumberOfPictures — number of vehicle pictures
# - PostalCode — postal code of profile owner (user)
# - LastSeen — date of the last activity of the user
# 
# Target
# - Price — price (Euro)
# 
# Analysis done January 2022

# ## Data preparation

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

import random
random_state=42
random.seed(random_state)
np.random.seed(random_state)

# import sys and insert code to ignore warnings 
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# __Helper functions__

# In[2]:


# function for timing execution of cell [refrence](https://stackoverflow.com/questions/52738709/how-to-store-time-values-in-a-variable-in-jupyter)
# couldn't get it to work though...
def exec_time(start, end): 
   diff_time = end - start
   m, s = divmod(diff_time, 60)
   h, m = divmod(m, 60)
   s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
   print("time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))

# function for displaying outlier statistics for column
def outlier_stats(data):
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    outliers_removed = [x for x in data if x >= lower and x <= upper]   
    outliers_stats = pd.Series(outliers)
    return outliers_stats.describe()

# function for displaying count and percent of missing values if column object or int64
def missing_stats(df):
    print("\nColumns Missing Values")
    df_missing = pd.concat([df.dtypes, df.isnull().sum(), df.isnull().sum() / (len(df)*.01)], axis=1)
    df_missing.columns = ['type', 'cnt', 'pct']
    print(df_missing)
    df.describe(include=['object', 'int64'])


# In[3]:


# load data
df = pd.read_csv('/datasets/car_data.csv')


# In[4]:


# inspect data
df.head()


# In[5]:


original_no_rows = df.shape[0]
df.shape


# In[6]:


df.duplicated().sum()


# In[7]:


missing_stats(df)


# __Rename columns to lowercase for consistency__

# In[8]:


df = df.rename(columns={'DateCrawled': 'date_crawled', 'Price': 'price', 'VehicleType': 'vehicle_type', 
                        'RegistrationYear': 'registration_year', 'Gearbox': 'gearbox', 'Power': 'power', 
                        'Model': 'model','Mileage': 'milage', 'RegistrationMonth': 'registration_month', 
                        'FuelType': 'fuel_type', 'Brand': 'brand','NotRepaired': 'not_repaired', 
                        'DateCreated': 'date_created', 'NumberOfPictures': 'num_pictures', 
                        'PostalCode': 'postal_code', 'LastSeen': 'last_seen', })

df.columns


# __Initial findings__
# 
# - 354369 rows by 16 columns/features, all listed as int64 or object
# - 262 duplicate rows --> delete
# - 5 features have missing rows --> investigate
# - Not all features will be useful in analysis --> investigate
# - Some dates are listed as objects --> chg datatype or delete feature

# In[9]:


# drop duplicate rows
df = df.drop_duplicates() 
df.shape


# In[10]:


# investigate columns
df.describe(include=['object', 'int64'])


# __Initial observations__ 
# - date_crawled, date_created and last_seen indicate these listing are from 2016
# - The latest year possible in registration_year is 2016 --> delete rows > 2016
# - date_crawled, date_created and last_seen are not useful for analysis --> delete columns
# - num_pictures only has 0.0, not useful for analysis --> delete column
# - postal_code has objects less than 5 digits long --> investigate and delete rows or postal_code column

# In[11]:


df.query('registration_year > 2016')


# In[12]:


14529/354369


# In[13]:


df.query('registration_year < 1959')


# In[14]:


390/354369


# __Since it is unlikely that we have many legitimate cars with registration dates less then 1960, we will eliminate those rows along with the rows of vehicles with a registration year > 2016, which would be impossible since the data was collected in 2016.__

# In[15]:


df = df.loc[df['registration_year'] <= 2016]
df = df.loc[df['registration_year'] >1959]
df.query('registration_year > 2016' and 'registration_year < 1960')


# In[16]:


df_bad_code = df.query('postal_code <10000') 
print('Pct of rows with postal_codes < 10000', df_bad_code.shape[0]/df.shape[0])


# In[17]:


df.corr()


# __We note a little over half a percent of the rows have an obviously inaccurate postal code so deleting those rows is one choice. However, the correlation between price and postal_code is fairly low, so we will drop the postal_code column instead__

# In[18]:


print(df.shape)
df.drop(columns=['date_crawled', 'date_created', 'last_seen', 'num_pictures', 'postal_code'], inplace=True)
df.shape


# In[19]:


missing_stats(df)


# __Observations with updated list__
# - price has outliers --> investigate outliers
# - vehicle_type has missing values --> fillin related to model
# - registration_year --> handled earlier
# - gearbox has missing values --> investigate and fillin 
# - power values vary widely --> [research online](https://www.autolist.com/guides/average-car-horsepower), investigate
# - __model has missing values --> investigate (first since will use for fill ins)__
# - milage no missing values, given values look reasonable --> keep as is
# - registration_month has some values of zero --> investigate
# - fuel_type, not sure how there are 7 unique values --> investigate
# - brand (make) has no missing values --> keep as is
# - not_repaired indicates if vehicle repaired or not --> investigate

# In[20]:


df['model'].value_counts(dropna=False)


# In[21]:


df['model'].isnull().sum()/len(df)


# In[22]:


df.loc[df['model'].isnull()].head()


# __model has 251 unique values and a little over half a percent are null. It is unlikely we can make wise replacements for the missing values, so we will remove those rows with missing the model value.__

# In[23]:


print(df.shape)
df.dropna(subset=['model'], inplace=True) 
df.shape               


# In[24]:


df['model'].isnull().sum()


# In[25]:


outlier_stats(df.price)


# In[26]:


df.query('price == 0')


# __There are almost 8000 rows where the price is 0. This may be clerical error or it may reflect vehicles junked or given for free. Either way, these will not be useful for our analysis. We will keep the rows with higher outlier values as some cars may have gone for an impressive amount.__

# In[27]:


df = df.loc[df['price'] != 0]
df.query('price == 0')


# In[28]:


df['vehicle_type'].value_counts(dropna=False)


# In[29]:


df['vehicle_type'].fillna(df.groupby('model')['vehicle_type'].
                          transform(lambda x:x.value_counts().index[0]), inplace=True)
df['vehicle_type'].value_counts(dropna=False)


# In[30]:


df['gearbox'].value_counts(dropna=False)


# __gearbox is missing about 4% of the values and we will replace those based on model since the gearbox is usally consistent with models.__ 

# In[31]:


df['gearbox'].fillna(df.groupby('model')['gearbox']                          .transform(lambda x:x.value_counts().index[0]), inplace=True)
df['gearbox'].value_counts(dropna=False)


# In[32]:


outlier_stats(df.power)


# __We discover a small number of outliers in the power column and will delete those.__

# In[33]:


print(df.shape)
df = df.loc[df['power'] <= 670]
df.shape 


# In[34]:


df['power'].isnull().sum()


# In[35]:


df['registration_month'].value_counts()


# In[36]:


df.corr()


# __Once again we find a weak relationship between registration_month and price, but it would be better to keep the column and rows for overall data integrity. The months are fairly evenly divided so we could randomly distribute the NA values with values of 1:12 to indicate months.__

# In[37]:


df['registration_month'] = df['registration_month'].apply(lambda v: random.choice([1,2,3,4,5,6,7,8,9,10,11,12]))
df['registration_month'].value_counts(dropna=False)


# In[38]:


df.query('registration_month == 0')


# In[39]:


df['fuel_type'].value_counts(dropna=False)


# __We will fill in the missing values using the influence of model since usually the fuel type is consistent with model.__

# In[40]:


df['fuel_type'].fillna(df.groupby('model')['fuel_type']                          .transform(lambda x:x.value_counts().index[0]), inplace=True)
df['fuel_type'].value_counts(dropna=False)


# In[41]:


df['not_repaired'].value_counts(dropna=False)


# __Since no is the most common value, we will replace the missing values with no.__

# In[42]:


df['not_repaired'].fillna('no', inplace=True)
df['not_repaired'].value_counts(dropna=False)


# __Now we will check to for any missed missing values, reset the index, and look at the percentage of the original data we deleted.__

# In[43]:


missing_stats(df)


# In[44]:


df.head()


# In[45]:


# resetting the DataFrame index
df = df.reset_index()
df.head(5)


# In[46]:


len(df)/original_no_rows


# __Summary of data preparation__
# - We dropped several columns, 'date_crawled', 'date_created', 'last_seen', 'num_pictures', 'postal_code', that will not be useful for our analysis.
# - We eliminated rows where the registration_year > 2016 (the year of the data) or < 1960 (unlikely).
# - We eliminated rows where the price = 0.
# - We eliminated the rows where a value for model was missing.
# - We filled in missing rows in vehicle_type, gearbox, fuel_type, power based on model.
# - We changed the zero values in registration_month to a randomly assigned # (1-12).
# - We replaced the NaN values in not_repaired with 'no'.
# - We verified that no missing values remain.
# - We note our preparation eliminated almost 12% of the data, but feel confident in the deletion choices.

# ## Model training

# In[47]:


# change into categories for lightBGM and CatBoost
categories = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand', 'not_repaired']
for col in categories:
    df[col] = df[col].astype('category')


# In[48]:


# create feature and target variables
target = df['price']
features = df.drop(['price'], axis=1)


# In[49]:


# use one hot encoding to turn the categories into numeric
features_ohe = pd.get_dummies(features, drop_first=True)


# In[50]:


# divide data with OHE into 3 groups using 3:1:1 (60%, 20%, 20%) ratio
features_train, features_valid, target_train, target_valid = train_test_split(
    features_ohe, target, test_size=0.4, random_state = 12345)
features_valid, features_test, target_valid, target_test = train_test_split(
    features_valid, target_valid, test_size=0.5, shuffle = False)
print('Train target and features and percentage\n', target_train.shape, features_train.shape,
      'pct', (len(target_train)/len(df)))
print('Valid target and features and percentage\n', target_valid.shape, features_valid.shape,
      'pct', (len(target_valid)/len(df)))
print('Test target and features and percentage\n', target_test.shape, features_test.shape,
      'pct', (len(target_test)/len(df)))


# In[51]:


# divide data without OHE into 3 groups using 3:1:1 (60%, 20%, 20%) ratio
features_train_2, features_valid_2, target_train_2, target_valid_2 = train_test_split(
    features, target, test_size=0.4, random_state = 12345)
features_valid_2, features_test_2, target_valid_2, target_test_2 = train_test_split(
    features_valid_2, target_valid_2, test_size=0.5, shuffle = False)

print('Train target and features and percentage\n', target_train_2.shape, features_train_2.shape,
      'pct', (len(target_train_2)/len(df)))
print('Valid target and features and percentage\n', target_valid_2.shape, features_valid_2.shape,
      'pct', (len(target_valid_2)/len(df)))
print('Test target and features and percentage\n', target_test_2.shape, features_test_2.shape,
      'pct', (len(target_test_2)/len(df)))


# In[52]:


# set up rmse calculation
def find_rmse(target_test, predictions):
    return round(mean_squared_error(target_test, predictions) ** 0.5, 2)
rmse = make_scorer(find_rmse, greater_is_better=False)


# __Note on timing cells. We used %%time to find the total time elapsed in each cell. However, we wanted to save the time values in variables and discovered %%time does not allow that. Therefore, we manually timed training time and prediction time and took note that their total matched closely with the %%time.__
# 
# __Looking at base models__

# In[53]:


get_ipython().run_cell_magic('time', '', "# linear regression with default parameters\nlr_model = LinearRegression()\n\nstart = time.time()\nlr_model.fit(features_train, target_train)\nend = time.time()\nlrtt = end - start\n\nstart = time.time()\npredicted_valid = lr_model.predict(features_valid)\nend = time.time()\nlrpt = end - start\n\nlr_rmse_calc = mean_squared_error(target_valid, predicted_valid)**0.5\n\nprint('Linear Regression - Sanity Check')\nprint('RMSE:', lr_rmse_calc, 'Training time:', lrtt, 'Prediction time:', lrpt)")


# In[54]:


get_ipython().run_cell_magic('time', '', "# random forest regressor with default parameters\nrf_model = RandomForestRegressor(random_state=42)\n\nstart = time.time()\nrf_model.fit(features_train, target_train)\nend = time.time()\nrftt = end - start\n\nstart = time.time()\npredicted_valid = rf_model.predict(features_valid)\nend = time.time()\nrfpt = end - start\n\nrf_rmse_calc = mean_squared_error(target_valid, predicted_valid)**0.5\n\nprint('Random Forest Regressor')\nprint('RMSE:', rf_rmse_calc, 'Training time:', rftt, 'Prediction time:', rfpt)")


# In[55]:


get_ipython().run_cell_magic('time', '', "# lightGBM with OHE \nlg_model = lgb.LGBMRegressor(random_state=42)\n\nstart = time.time()\nlg_model.fit(features_train, target_train)\nend = time.time()\nlgohett = end - start\n\nstart = time.time()\npredicted_valid = lg_model.predict(features_valid)\nend = time.time()\nlgohept = end - start\n\nlgohe_rmse_calc = mean_squared_error(target_valid, predicted_valid)**0.5\n\nprint('LightGBM with OHE')\nprint('RMSE:', lgohe_rmse_calc, 'Training time:', lgohett, 'Prediction time:', lgohept)")


# In[56]:


get_ipython().run_cell_magic('time', '', "# lightGBM without OHE \nlg_model_2 = lgb.LGBMRegressor(random_state=42)\n\nstart = time.time()\nlg_model_2.fit(features_train_2, target_train_2, categorical_feature=categories)\nend = time.time()\nlgtt = end - start\n\nstart = time.time()\npredicted_valid = lg_model_2.predict(features_valid_2)\nend = time.time()\nlgpt = end - start\n\nlg_rmse_calc = mean_squared_error(target_valid_2, predicted_valid)**0.5\n\nprint('LightGBM without OHE')\nprint('RMSE:', lg_rmse_calc, 'Training time:', lgtt, 'Prediction time:', lgpt)")


# In[57]:


get_ipython().run_cell_magic('time', '', "# CatBoost with OHE \ncb_model = CatBoostRegressor(random_state=42)\n\nstart = time.time()\ncb_model.fit(features_train, target_train)\nend = time.time()\ncbohett = end - start\n\nstart = time.time()\npredicted_valid = cb_model.predict(features_valid)\nend = time.time()\ncbohept = end - start\n\ncbohe_rmse_calc = mean_squared_error(target_valid, predicted_valid)**0.5\n\nprint('CatBoost with OHE')\nprint('RMSE:', cbohe_rmse_calc, 'Training time:', cbohett, 'Prediction time:', cbohept)")


# In[58]:


get_ipython().run_cell_magic('time', '', "# CatBoost without OHE \ncb_model = CatBoostRegressor(random_state=42)\n\nstart = time.time()\ncb_model.fit(features_train_2, target_train_2, cat_features=categories)\nend = time.time()\ncbtt = end - start\n\nstart = time.time()\npredicted_valid = cb_model.predict(features_valid_2)\nend = time.time()\ncbpt = end - start\n\ncb_rmse_calc = mean_squared_error(target_valid_2, predicted_valid)**0.5\n\nprint('CatBoost without OHE')\nprint('RMSE:', cb_rmse_calc, 'Training time:', cbtt, 'Prediction time:', cbpt)")


# In[59]:


print('Linear Regression - Sanity Check')
print('RMSE:', lr_rmse_calc, 'Training time:', lrtt, 'Prediction time:', lrpt)
print('\nRandom Forest Regressor')
print('RMSE:', rf_rmse_calc, 'Training time:', rftt, 'Prediction time:', rfpt)
print('\nLightGBM with OHE')
print('RMSE:', lgohe_rmse_calc, 'Training time:', lgohett, 'Prediction time:', lgohept)
print('\nLightGBM without OHE')
print('RMSE:', lg_rmse_calc, 'Training time:', lgtt, 'Prediction time:', lgpt)
print('\nCatBoost with OHE')
print('RMSE:', cbohe_rmse_calc, 'Training time:', cbohett, 'Prediction time:', cbohept)
print('\nCatBoost without OHE')
print('RMSE:', cb_rmse_calc, 'Training time:', cbtt, 'Prediction time:', cbpt)


# __We note the RMSE and the times required for training and prediction__
# 
# - All the models perform better on RMSE (goodness of fit) than linear regression, which we expected.
# - LightGBM without One Hot Encoding provides the best RMSE, but all the values besides linear regression are pretty close.
# - The training times of the base models range from around 20 seconds with LightGBM to just over 500 seconds with CatBoost.
# - The prediction times of the base models range from around .2 seconds (CatBoost) to close to 1.2 with LightGBM w/OHE.
# - Since the model needs to be trained once, the prediction time is actually more relevant for running multiple predictions. 

# __Tuning hyperparameters__
# 
# - We will look at n_estimators with RandomForestRegressor
# - We will tune parameters of CatBoost and LightGBM using both with and without OHE since there is variable performance [LightGBMdoc](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html), [CatBoost](https://catboost.ai/)

# In[60]:


get_ipython().run_cell_magic('time', '', "rf_model = RandomForestRegressor(random_state=42)\nparams = { 'n_estimators': range(10, 30, 5) }\n\nbest_model = RandomizedSearchCV(rf_model, params, scoring=rmse, cv=5, verbose=10)\nbest_model.fit(features_train, target_train)  \nprint('Best parameters:', best_model.best_params_)\n\npredictions = best_model.best_estimator_.predict(features_valid)\nprint('RMSE:', round(mean_squared_error(target_valid, predictions) ** 0.5, 2))")


# __Tuning for Random Forest Regressor__
# 
# __Print and copy results so we don't run use another 30 minute block by running the code__
# 
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# [CV] n_estimators=10 .................................................
# [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
# [CV] ................. n_estimators=10, score=-1772.600, total=  49.3s
# [CV] n_estimators=10 .................................................
# [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:   49.3s remaining:    0.0s
# [CV] ................. n_estimators=10, score=-1711.530, total=  49.8s
# [CV] n_estimators=10 .................................................
# [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:  1.7min remaining:    0.0s
# [CV] ................. n_estimators=10, score=-1736.270, total=  49.9s
# [CV] n_estimators=10 .................................................
# [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:  2.5min remaining:    0.0s
# [CV] ................. n_estimators=10, score=-1717.170, total=  49.5s
# [CV] n_estimators=10 .................................................
# [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:  3.3min remaining:    0.0s
# [CV] ................. n_estimators=10, score=-1741.730, total=  48.3s
# [CV] n_estimators=15 .................................................
# [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:  4.1min remaining:    0.0s
# [CV] ................. n_estimators=15, score=-1744.790, total= 1.2min
# [CV] n_estimators=15 .................................................
# [Parallel(n_jobs=1)]: Done   6 out of   6 | elapsed:  5.3min remaining:    0.0s
# [CV] ................. n_estimators=15, score=-1686.140, total= 1.2min
# [CV] n_estimators=15 .................................................
# [Parallel(n_jobs=1)]: Done   7 out of   7 | elapsed:  6.6min remaining:    0.0s
# [CV] ................. n_estimators=15, score=-1710.780, total= 1.2min
# [CV] n_estimators=15 .................................................
# [Parallel(n_jobs=1)]: Done   8 out of   8 | elapsed:  7.7min remaining:    0.0s
# [CV] ................. n_estimators=15, score=-1692.710, total= 1.2min
# [CV] n_estimators=15 .................................................
# [Parallel(n_jobs=1)]: Done   9 out of   9 | elapsed:  8.9min remaining:    0.0s
# [CV] ................. n_estimators=15, score=-1719.410, total= 1.2min
# [CV] n_estimators=20 .................................................
# [CV] ................. n_estimators=20, score=-1729.780, total= 1.6min
# [CV] n_estimators=20 .................................................
# [CV] ................. n_estimators=20, score=-1676.970, total= 1.6min
# [CV] n_estimators=20 .................................................
# [CV] ................. n_estimators=20, score=-1704.250, total= 1.6min
# [CV] n_estimators=20 .................................................
# [CV] ................. n_estimators=20, score=-1678.980, total= 1.6min
# [CV] n_estimators=20 .................................................
# [CV] ................. n_estimators=20, score=-1708.880, total= 1.6min
# [CV] n_estimators=25 .................................................
# [CV] ................. n_estimators=25, score=-1722.470, total= 2.0min
# [CV] n_estimators=25 .................................................
# [CV] ................. n_estimators=25, score=-1668.740, total= 2.1min
# [CV] n_estimators=25 .................................................
# [CV] ................. n_estimators=25, score=-1694.250, total= 2.0min
# [CV] n_estimators=25 .................................................
# [CV] ................. n_estimators=25, score=-1669.860, total= 2.0min
# [CV] n_estimators=25 .................................................
# [CV] ................. n_estimators=25, score=-1700.940, total= 2.0min
# [Parallel(n_jobs=1)]: Done  20 out of  20 | elapsed: 28.1min finished
# Best parameters: {'n_estimators': 25}
# RMSE: 1681.06
# CPU times: user 30min 16s, sys: 2.41 s, total: 30min 19s
# Wall time: 30min 45s
# 
# Best parameters: {'n_estimators': 25}
# 
# RMSE: 1681.06
# 
# CPU times: user 30min 16s, sys: 2.41 s, total: 30min 19s
# 
# Wall time: 30min 45s

# In[73]:


get_ipython().run_cell_magic('time', '', "# CatBoost without OHE\ncb_model_2 = CatBoostRegressor(random_state=42)\nparams = { 'n_estimators': range(10, 30, 5), 'learning_rate': [.25, .5, .75] }\n\nbest_model = RandomizedSearchCV(cb_model_2, params, scoring=rmse, cv=5, verbose=10)\nbest_model.fit(features_train_2, target_train_2, cat_features=categories) \npredictions = best_model.best_estimator_.predict(features_valid_2)\n\nprint('CatBoost without OHE')\nprint('Best parameters:', best_model.best_params_)\nprint('RMSE:', round(mean_squared_error(target_valid_2, predictions) ** 0.5, 2))")


# __Tuning for CatBoost without OHE__
# 
# CatBoost without OHE
# 
# Best parameters: {'n_estimators': 25, 'learning_rate': 0.75}
# 
# RMSE: 1787.49
# 
# CPU times: user 3min 42s, sys: 27.7 s, total: 4min 10s
# 
# Wall time: 5min 8s

# In[74]:


get_ipython().run_cell_magic('time', '', "# CatBoost with OHE\ncb_model = CatBoostRegressor(random_state=42)\nparams = { 'n_estimators': range(10, 30, 5), 'learning_rate': [.25, .5, .75] }\n\nbest_model = RandomizedSearchCV(cb_model, params, scoring=rmse, cv=5, verbose=10)\nbest_model.fit(features_train, target_train) \npredictions = best_model.best_estimator_.predict(features_valid)\n\nprint('CatBoost with OHE')\nprint('Best parameters:', best_model.best_params_)\nprint('RMSE:', round(mean_squared_error(target_valid, predictions) ** 0.5, 2))")


# __Tuning for CatBoost with OHE__
# 
# CatBoost with OHE
# 
# Best parameters: {'n_estimators': 25, 'learning_rate': 0.75}
# 
# RMSE: 1814.42
# 
# CPU times: user 5min 24s, sys: 27.8 s, total: 5min 52s
# 
# Wall time: 6min 46s

# In[75]:


get_ipython().run_cell_magic('time', '', "# LIghtGBM with OHE\nlg_model = lgb.LGBMRegressor(random_state=42)\nparams = { 'n_estimators': range(10, 30, 5), 'learning_rate': [.25, .5, .75] }\n\nbest_model = RandomizedSearchCV(lg_model, params, scoring=rmse, cv=5, verbose=10)\nbest_model.fit(features_train, target_train) \npredictions = best_model.best_estimator_.predict(features_valid)\n\nprint('LightGBM with OHE')\nprint('Best parameters:', best_model.best_params_)\nprint('RMSE:', round(mean_squared_error(target_valid, predictions) ** 0.5, 2))")


# __Tuning for LightGBM with OHE__
# 
# LightGBM with OHE
# 
# Best parameters: {'n_estimators': 25, 'learning_rate': 0.5}
# 
# RMSE: 1731.08
# 
# CPU times: user 3min 14s, sys: 14.1 s, total: 3min 28s
# 
# Wall time: 3min 30s
# 

# In[76]:


get_ipython().run_cell_magic('time', '', "# LightGBM without OHE\nlg_model_2 = lgb.LGBMRegressor(random_state=42)\nparams = { 'n_estimators': range(10, 30, 5), 'learning_rate': [.25, .5, .75] }\n\nbest_model = RandomizedSearchCV(lg_model_2, params, scoring=rmse, cv=5, verbose=10)\nbest_model.fit(features_train_2, target_train_2, categorical_feature=categories)  \npredictions = best_model.best_estimator_.predict(features_valid_2)\n\nprint('LightGBM without OHE')\nprint('Best parameters:', best_model.best_params_)\nprint('RMSE:', round(mean_squared_error(target_valid_2, predictions) ** 0.5, 2))")


# __Tuning for LightGBM without OHE__
# 
# LightGBM without OHE
# 
# Best parameters: {'n_estimators': 25, 'learning_rate': 0.25}
# 
# RMSE: 1693.92
# 
# CPU times: user 4min 59s, sys: 960 ms, total: 5min
# 
# Wall time: 5min 3s

# __Fit models with best parameters__
# 
# - We will use the combined data from the train and valid datasets because that gives us a slightly better RMSE score (we compared previously).
# - We will compare tuned and untuned versions.

# In[77]:


get_ipython().run_cell_magic('time', '', "# random forest regressor with test data\n# Best parameters: {'n_estimators': 25}\nrf_test = RandomForestRegressor(random_state=42, n_estimators = 25)\n\nstart = time.time()\n# fit with train and valid data\nrf_test.fit(pd.concat([features_train, features_valid]), pd.concat([target_train, target_valid]))\nend = time.time()\ntest_rftt = end - start\n\nstart = time.time()\ntest_pred = rf_test.predict(features_test)\nend = time.time()\ntest_rfpt = end - start\n\ntest_rf_rmse_calc = mean_squared_error(target_test, test_pred)**0.5\n\nprint('Random Forest Regressor with test data')\nprint('RMSE:', test_rf_rmse_calc, 'Training time:', test_rftt, 'Prediction time:', test_rfpt)")


# In[78]:


get_ipython().run_cell_magic('time', '', "# CatBoost with OHE with test data\n# Best parameters: {'n_estimators': 25, 'learning_rate': 0.75}\ncb_test = CatBoostRegressor(random_state=42, n_estimators=25, learning_rate=0.75)\n\nstart = time.time()\n# use combined train and valid datasets to fit on\ncb_test.fit(pd.concat([features_train, features_valid]), pd.concat([target_train, target_valid]))\nend = time.time()\ntest_cbohett = end - start\n\nstart = time.time()\ntest_pred = cb_test.predict(features_test)\nend = time.time()\ntest_cbohept = end - start\n\ntest_cbohe_rmse_calc = mean_squared_error(target_test, test_pred)**0.5\n\nprint('CatBoost with OHE with test data')\nprint('RMSE:', test_cbohe_rmse_calc, 'Training time:', test_cbohett, 'Prediction time:', test_cbohept)")


# In[79]:


get_ipython().run_cell_magic('time', '', "# CatBoost without initial one hot encoding\n# Best parameters: {'n_estimators': 25, 'learning_rate': 0.75}\n\nstart = time.time()\n# use combined train and valid datasets to fit on\ncb_test_2 = CatBoostRegressor(random_state=42, n_estimators=25, learning_rate=0.75)\ncb_test_2.fit(pd.concat([features_train_2, features_valid_2]), pd.concat([target_train_2, target_valid_2]), cat_features=categories)\nend = time.time()\ntest_cbtt = end - start\n\nstart = time.time()\ntest_pred = cb_test_2.predict(features_test_2)\nend = time.time()\ntest_cbpt = end - start\n\ntest_cb_rmse=mean_squared_error(target_test, test_pred)**0.5\n\nprint('CatBoost without OHE with test data')\nprint('RMSE:',test_cb_rmse, 'Training time:', test_cbtt, 'Prediction time:', test_cbpt) ")


# In[80]:


get_ipython().run_cell_magic('time', '', "# lightGBM with OHE with test data\n# Best parameters: {'n_estimators': 25, 'learning_rate': 0.5}\nlgohe_test = lgb.LGBMRegressor(random_state=42, n_estimators=25, learning_rate=0.5)\n\nstart = time.time()\n# fit model on combined data from train and valid\nlgohe_test.fit(pd.concat([features_train, features_valid]), pd.concat([target_train, target_valid]))\nend = time.time()\ntest_lgohett = end - start\n\nstart = time.time()\ntest_pred = lgohe_test.predict(features_test)\nend = time.time()\ntest_lgohept = end - start\n\ntest_lgohe_rmse_calc = mean_squared_error(target_test, test_pred)**0.5\n\nprint('LightGBM with OHE with test data')\nprint('RMSE:', test_lgohe_rmse_calc, 'Training time:', test_lgohett, 'Prediction time:', test_lgohept)")


# In[81]:


get_ipython().run_cell_magic('time', '', "# lightGBM without OHE with test data\n# Best parameters: {'n_estimators': 25, 'learning_rate': 0.25}\nlg_test = lgb.LGBMRegressor(random_state=42, n_estimators=25, learning_rate=0.25)\n\nstart = time.time()\n# fit on combined data from train and valid\nlg_test.fit(pd.concat([features_train_2, features_valid_2]), pd.concat([target_train_2, target_valid_2]), categorical_feature=categories)\nend = time.time()\ntest_lgtt = end - start\n\nstart = time.time()\ntest_pred = lg_test.predict(features_test_2)\nend = time.time()\ntest_lgpt = end - start\n\ntest_lg_rmse_calc = mean_squared_error(target_test_2, test_pred)**0.5\n\nprint('LightGBM without OHE')\nprint('RMSE:', test_lg_rmse_calc, 'Training time:', test_lgtt, 'Prediction time:', test_lgpt)")


# In[82]:


# linear regression base model with test data
start = time.time()
predicted_test = lr_model.predict(features_test)
end = time.time()
test_lrpt = end - start

test_lr_rmse_calc = mean_squared_error(target_test, predicted_test)**0.5

print('Linear Regression - Sanity Check on test data')
print('RMSE:', test_lr_rmse_calc, 'Prediction time:', test_lrpt)


# ## Model analysis

# In[83]:


print('Results with test data of base models\n')

print('Random Forest Regressor')
print('RMSE:', rf_rmse_calc, 'Training time:', rftt, 'Prediction time:', rfpt)
print('\nLightGBM with OHE')
print('RMSE:', lgohe_rmse_calc, 'Training time:', lgohett, 'Prediction time:', lgohept)
print('\nLightGBM without OHE')
print('RMSE:', lg_rmse_calc, 'Training time:', lgtt, 'Prediction time:', lgpt)
print('\nCatBoost with OHE')
print('RMSE:', cbohe_rmse_calc, 'Training time:', cbohett, 'Prediction time:', cbohept)
print('\nCatBoost without OHE')
print('RMSE:', cb_rmse_calc, 'Training time:', cbtt, 'Prediction time:', cbpt)
print('\nLinear Regression - Sanity Check')
print('RMSE:', lr_rmse_calc, 'Training time:', lrtt, 'Prediction time:', lrpt)


# In[84]:


print('Results with test data of tuned models\n')

print('Random Forest Regressor with test data')
print('RMSE:', test_rf_rmse_calc, 'Training time:', test_rftt, 'Prediction time:', test_rfpt)

print('\nCatBoost with OHE with test data')
print('RMSE:', test_cbohe_rmse_calc, 'Training time:', test_cbohett, 'Prediction time:', test_cbohept)

print('\nCatBoost without OHE with test data')
print('RMSE:',test_cb_rmse, 'Training time:', test_cbtt, 'Prediction time:', test_cbpt)

print('\nLightGBM with OHE with test data')
print('RMSE:', test_lgohe_rmse_calc, 'Training time:', test_lgohett, 'Prediction time:', test_lgohept)

print('\nLightGBM without OHE')
print('RMSE:', test_lg_rmse_calc, 'Training time:', test_lgtt, 'Prediction time:', test_lgpt)

print('\nLinear Regression - Sanity Check on test data')
print('RMSE:', test_lr_rmse_calc, 'Prediction time:', test_lrpt)


# __We were tasked with finding the RMSE, the training time and the prediction time__
# 
# 
# - Tuning the Random Forest Regressor model did improve the goodness of fit as evidenced by a lower RMSE, but tuning also more than doubled the time for predictions and training.
# - The RMSE for CatBoost and LightGBM did not benefit from my implementation of performance tuning.
# - However, tuning the parameters decreased the prediction time dramatically.
# - The best RMSE for both tuned and untuned models is LightGBM without OHE. This matches nicely with [documentation of LightGBM](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html). It states LightGBM "offers good accuracy with integer-encoded categorical features. LightGBM applies Fisher (1958) to find the optimal split over categories as described here. This often performs better than one-hot encoding."
# - LightGBM, both with and without OHE, had the fastest prediction speeds. CatBoost had the slowest.
# - LightGBM had lower prediction times, and tuning the model lowered them further.
# - While the tuned Random Forest Regressor has the lowest RMSE, it also has the highest prediction time, over double of the second longest. The slight gains in RMSE over LIghtGBM without OHE doesn't compensate for the large increase in prediction time.
# 
# __Recommendation__
# 
# - We recommend the tuned LightGBM without One Hot Encoding. It has the 3rd lowest RMSE scores, but also has a fast training time, just over 11 seconds, and an impressive prediction time of about 1/5th of a second.

# [Timing cells](https://stackoverflow.com/questions/52738709/how-to-store-time-values-in-a-variable-in-jupyter)
# 
# [simple function to time execution of cell](https://stackoverflow.com/questions/52738709/how-to-store-time-values-in-a-variable-in-jupyter)
# 
# [LightGBM often performs better without OHE](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html)
# 

# In[ ]:





# In[ ]:




