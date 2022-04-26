#!/usr/bin/env python
# coding: utf-8

# # Analyzing borrowers’ risk of defaulting
# 
# Your project is to prepare a report for a bank’s loan division. You’ll need to find out if a customer’s marital status and number of children has an impact on whether they will default on a loan. The bank already has some data on customers’ credit worthiness.
# 
# Your report will be considered when building a **credit scoring** of a potential customer. A ** credit scoring ** is used to evaluate the ability of a potential borrower to repay their loan.

# In[1]:


# import libraries
import pandas as pd
import numpy as np


# In[2]:


# load the data 
try:
    data = pd.read_csv('/Users/rraven/Desktop/a_final_yandex/datasets/credit_scoring_eng.csv')
except:
    data = pd.read_csv('/datasets/credit_scoring_eng.csv')


# In[3]:


# print general info & first 10 rows
#data = pd.read_csv('../data/credit_scoring_eng.csv')
print('\nGeneral Info for credit_scoring_eng\n')
data.info()
print('\nFirst 10 rows of credit_scoring_eng.csv')
data.head(10)


# In[4]:


# find the number of missing values per column
print('Total Rows:', len(data))
print('\nColumn\t\tMissing Rows')
data.isnull().sum()


# In[5]:


ratio = 100*data.isnull().sum() / len(data)
print('Column\t\tPercent Missing')
ratio


# In[6]:


# check data for duplicates in entire dataFrame
print('Number of duplicate rows:')
data.duplicated().sum()


# **Initial Observations for credit_scoring_eng.csv**
# 
# * 21525 rows
# * 12 columns
# * A mix of data types (float64(2), int64(5), object(5))
# * 2174 missing values in 2 columns (days_employed, total_income) 
# * days_employed and total_income are missing over 10% of total values
# * 54 duplicate rows

# In[7]:


# investigate children column
data['children'].value_counts()


# In[8]:


# investigate family_status column
data['family_status'].value_counts()


# In[9]:


# investigate family_status_id column
data['family_status_id'].value_counts()


# In[10]:


# investigate purpose column
data['purpose'].value_counts()


# In[11]:


# how many unique entries in purpose column
data['purpose'].nunique()


# In[12]:


# investigate debt column
data['debt'].value_counts()


# In[13]:


# investigate dob_years column
data['dob_years'].value_counts().sort_index()


# In[14]:


# categorize dob_years in 10 year increments
less_than_20 = 0
btwn20and30 = 0
btwn30and40 = 0
btwn40and50 = 0
btwn50and60 = 0
btwn60and70 = 0
over_equal70 = 0
nan_count = 0
unknown = 0
for x in data['dob_years']:
    if x < 20:
        less_than_20 +=1
    elif x < 30:
        btwn20and30 +=1
    elif x < 40:
        btwn30and40 +=1   
    elif x < 50:
        btwn40and50 +=1 
    elif x < 60:
        btwn50and60 +=1 
    elif x < 70:
        btwn60and70 +=1 
    elif x >= 70:
        over_equal70 +=1        
    elif str(x) == 'nan':
        nan_count +=1
    else:
        # to check there are no other values
        unknown +=1
print("Age < 20:\t", less_than_20, 
      "\n20 <= Age < 30:\t", btwn20and30,
      "\n30 <= Age < 40:\t", btwn30and40,
      "\n40 <= Age < 50:\t", btwn40and50,
      "\n50 <= Age < 60:\t", btwn50and60,
      "\n60 <= Age < 70:\t", btwn60and70,
      "\nAge >= 70:\t", over_equal70,
      "\nnan:\t\t", nan_count,
      "\nUnknown:\t", unknown)


# In[15]:


# investigate education column
data['education'].value_counts()


# In[16]:


# investigate education_id column
data['education_id'].value_counts()


# In[17]:


# investigate gender column
data['gender'].value_counts()


# In[18]:


# investigate income_type column
data['income_type'].value_counts()


# In[19]:


# how many unique entries in income_type column
data['income_type'].nunique()


# In[20]:


# investigate total_income
data['total_income'].value_counts()


# In[21]:


# categorize income_level into 10K levels to see distribution

# create a new function, income_level_fx
def income_level_fx(row):
    # the income_level is returned according to total_income
    income = row['total_income']
    
    if income < 10000:
        return 'less_than_10K' 
    elif income < 20000:
        return 'btwn10Kand20K'
    elif income < 30000:
        return 'btwn20Kand30K'   
    elif income < 40000:
        return 'btwn30Kand40K'  
    elif income < 50:
        return 'btwn40Kand50K' 
    elif income < 60000:
        return 'btwn50Kand60K'  
    elif income < 70000:
        return 'btwn60Kand70K' 
    elif income < 80000:
        return 'btwn70Kand80K' 
    elif income < 90000:
        return 'btwn80Kand90K' 
    elif income < 100000:
        return 'btwn90Kand100K'
    elif income >= 100000:
        return 'greater_than_100K'  
    
# create a new column, income_level, based on total_income
data['income_level'] = data.apply(income_level_fx, axis=1)

# verfiy new column: income_level
data['income_level'].value_counts()


# ### Conclusion

# **The datafile credit_scoring_eng.csv contains**
# * 21525 rows
# * 12 columns
# * A mix of data types (float64(2), int64(5), object(5))
# * 2174 missing values in 2 columns (days_employed, total_income) 
# * days_employed and total_income are missing over 10% of total values
# * 54 duplicate rows
# 
# **Targeted areas - having kids, marital status, income level and loan purpose r/t repayment**
# * children 
#     - int64
#     - contains 47 entries with -1 and 76 with 20
#     - maybe -1 was meant to be 1? keying error? correct or remove?
#     - maybe 20 was meant to be 2? keying error? correct or remove?
#     - ultimately can try different groups (none or some), (none, 1, or > 1), etc
# * family_status and family_status_id 
#     - string(object) and int64
#     - no problems noted, appear to correlate well
#         - married              12380 
#         - civil partnership     4177
#         - unmarried             2813
#         - divorced              1195
#         - widow / widower        960
#         - 0    12380
#         - 1     4177
#         - 4     2813
#         - 3     1195
#         - 2      960
# * total_income
#     - float64
#     - will need to address missing values    
# * purpose
#     - string(object)
#     - 38 unique entries
#     - appears to be 4 main categories 
#         - wedding
#         - property
#         - car
#         - education
#     - will need to work with stem and/or lemmas
#     
# **How to judge risk of default / likelihood of repayment of loan?**
# Discussion: There are 12 total columns. 5 of these are specifically involved in the task (children, family_status/family_status_id, total_income, and purpose). This leaves days_employed, dob_years, education,
# education_id, gender, income_type, and debt.
# 
# Which of these are likely to correlate with risk of default? 
# - days_employed might, because it could demonstrate stability, but the values in that column have too many problems to be useful. Therefore, that column will likely be ignored. 
# - dob_years: could younger people (< 30 or 40) have a higher risk of default? maybe
# - education / education_id: could less education correlate with higher risk of default? maybe
# - gender: could gender correlate with risk of default? maybe
# - income_type: could source of income correlate with risk of default? maybe
# - debt: does "whether the customer has ever defaulted on a loan" correlate with risk of default? very likely
# 
# **Of these choices, debt is the most likely to produce useful correlations with risk of default. **
#   
# * debt
#     - int64
#     - 2 values, no nan or unknowns
#     - assume 0 hasn't ever defaulted 
#     - assume 1 for has defaulted in the past
#         - 0    19784
#         - 1     1741
#      
# **General observations of other columns**
# * days_employed (will investigate in missing info section)
#     - float64 (why?) 
#     - will need to address missing values     
# * dob_years
#     - int64
#     - age range from 19 to 75, but 101 values of 0 for age
#     - will need to categorize, maybe by decade 
# * education has a mixture of upper and lower case strings, numerous variations
#     - string(object)
#     - change all to lowercase to unify and compare to education_id
#     - could use it to help fill in missing income info
# * eduction_id column
#     - int64
#     - could be used instead of education if codes correlate?
#         - 0 = bachelor's degree
#         - 1 = secondary education
#         - 2 = some college
#         - 3 = primary education
#         - 4 = graduate degree
#     - maybe could reorganize because this order doesn't make sense?
# * gender
#     - string(object)
#     - could change it to number code
# * income_type
#     - string(object)
#     - 8 unique entries
#     - could change it to number code
#     - could combine it with other info to fill in missing incomes
#         - employee                       11119
#         - business                        5085
#         - retiree                         3856
#         - civil servant                   1459
# 

# ## Data preprocessing

# ### Processing missing values

# 1. Address days_employed
# 2. Address total_income

# In[22]:


# investigate days_employed column
data['days_employed'].value_counts().sort_index()


# In[23]:


# there are missing values, but how many?
# find percent of total rows where days_employed is missing a value
missing_days = data['days_employed'].isnull().sum()
pct_missing = missing_days/len(data)
print('Percentage of rows missing total_income: {:.2%}'.format(pct_missing))


# In[24]:


# add column & calculate years worked using days_employed
data['years_employed'] = data['days_employed'] / 365


# In[25]:


# investigate range of years_employed column
data['years_employed'].value_counts().sort_index()


# There are some unrealistic values (working for negative 50 years and over 1100 years).

# In[26]:


# find counts of positive and negative values for days employed
pos_count = 0
neg_count = 0
nan_count = 0
unknown = 0
for x in data['days_employed']:
    if x >= 0:
        pos_count +=1
    elif x < 0:
        neg_count +=1
    elif str(x) == 'nan':
        nan_count +=1
    else:
        # to check there are no other values
        unknown +=1
print("Positive days employed:\t", pos_count, 
      "\nNegative days employed:\t", neg_count,
      "\nnan in days employed:\t", nan_count,
      "\nUnknown in days employed:", unknown)


# Analysis of days_employed reveals:
# - float64 (why?) and has many negative values (why?)
# - is missing 2174 values
# - has a range from -18388.949901 to 401755.400475 days employed
# - has a range from -50 years to over 1100 years employed
# - positive days employed: 3445 
# - negative days employed: 15906 
# 
# Possible reasons for problem entries:
# 1. Human error when entering data (adding - by accident).
# 2. Error in units (perhaps the person entering the data got confused between hours worked and days worked?). Confusing hours for days may explain the very high numbers.
# 3. There may have been errors when merging different data sets.
# 
# Plan to fill missing values:
# 1. Since negative values are likely due to human error when entering data, use absolute value on days_employed.
# 2. Evaluate min/max values to determine range again, verify no more negative values.
# 3. Refresh years_employed and verify no more negative values.
# 4. Compare mean and median to gain an overall sense of data.
# 5. If people mistakenly entered hours worked instead of days, we can assume the maximum work period could be 70 years (70 yrs * 365 days/yrs) = 25550 days.
# 6. More realistically, we could assume the maximum work period could be 50 years (50 yrs * 365 days/yrs) = 18250.
# 7. With values greater than work period (either 25550 days or 18250 days), divide by 24 to get a days worked: data_50_yrs and data_70_years.
# 8. Compare statistics data_50_yrs and data_70_years.
# 9. Replace values in data df with appropriate cutoff value (#/24).
# 10. Compare mean and median of days_employed column.
# 11. Fill missing values with appropriate value (mean or median).

# In[27]:


# apply abs() to change negative values to positive
data['days_employed'] = data['days_employed'].abs()
# verify no more negative values after abs()
negative_count = data.loc[data['days_employed']  < 0, 'days_employed'].count()
print('After applying abs(), verify the total number of negative values in days_employed = 0.')
negative_count


# In[28]:


positive_count = data.loc[data['days_employed']  > 0, 'days_employed'].count()
print('After applying abs(), verify the total number of positive values in days_employed = 19351')
positive_count


# In[29]:


# add column & calculate years worked using days_employed
data['years_employed'] = data['days_employed'] / 365

days_neg = data.loc[data['years_employed']  < 0, 'years_employed'].count()
print('After applying abs(), verify the total number of negative values in years_employed = 0.')
days_neg


# In[30]:


# find info on mean, min, max for days_employed
print('Statistical info for days_employed')
data['days_employed'].describe()


# In[31]:


# find info on mean, min, max for years_employed
print('Statistical info for years_employed')
data['years_employed'].describe()


# In[32]:


# find the median of years_employed
data['years_employed'].median()


# This data is skewed with some very high values. The median for years is just over 6, while the mean is over 183 years worked. This suggests there are many very high values distorting the data.

# In[33]:


# make a copy of data, name it data_50_yrs
data_50_yrs = data.copy()
# change all the values greater than 18250 days (50 years) by dividing by 24 and saving results
data_50_yrs.loc[data_50_yrs['days_employed'] > 18250, 'days_employed'] = data_50_yrs['days_employed']/24
# find info on mean, min, max for days employed where values changed
print('Statistical info for days_employed')
print('where values > 18250 (50 years) and divided by 24')
data_50_yrs['days_employed'].describe()


# In[34]:


# make a copy of data, name it data_70_yrs
data_70_yrs = data.copy()
# change all the values greater than 25550 days (70 years) by dividing by 24 and saving results
data_70_yrs.loc[data_70_yrs['days_employed'] > 25550, 'days_employed'] = data_70_yrs['days_employed']/24
# find info on mean, min, max for days employed where values changed
print('Statistical info for days_employed')
print('where values > 25550 (70 years) and divided by 24')
data_70_yrs['days_employed'].describe()


# The means and medians for the two options are very close:
# 
# - 50 yrs  70 yrs
# - 4640 vs 4641
# - 2194 vs 2194
# 
# The max values are a bit different
# 17615 (48 years) vs 18388 (50 years).
# 
# The 70 year cutoff will be used in the data df just in case that entry with a 50 year work history is accurate. There are 3800+ retirees in the sample after all.

# In[35]:


# change all the values greater than 25550 days (70 years) by dividing by 24 and saving results
data.loc[data['days_employed'] > 25550, 'days_employed'] = data['days_employed']/24

# find info on mean, min, max for days_employed
print('Statistical info for days_employed')
data['days_employed'].describe()


# In[36]:


# look for nan, negative, and other values in days_employed
# print first 20 rows of duplicates to look for patterns
pos_count = 0
neg_count = 0
nan_count = 0
unknown = 0
counter= -1
for x in data['days_employed']:
    counter +=1
    if x >= 0:
        pos_count +=1
    elif x < 0:
        neg_count +=1
    elif str(x) == 'nan':
        nan_count +=1
        if nan_count < 20:
            print(data.iloc[counter])
    else:
        # to check there are no other values
        unknown +=1
print("Positive income:", pos_count, 
      "\nNegative income:", neg_count,
      "\nnan:\t\t", nan_count,
      "\nUnknown:\t", unknown)


# In[37]:


# find median of days_employed
median_days = data['days_employed'].median()
print('The median of days_employed')
median_days


# In[38]:


# fill in missing values with median income
data['days_employed'].fillna(value=median_days, inplace = True)


# In[39]:


# check for any missing values
# find info on mean, min, max for days_employed
print('Statistical info for days_employed')
data['days_employed'].describe()


# There are no more missing values and the median remains the same, 2194. The mean actually went down, which makes sense because over 2000 missing values were replaced with 2194, bringing the overall mean downward.

# In[40]:


# find percent of total rows where total_income is missing
missing_income = data['total_income'].isnull().sum()
pct_missing = missing_income/len(data)
print('Percentage of rows missing total_income: {:.2%}'.format(pct_missing))


# In[41]:


# look for nan, negative, and other values in total_income
# print first 20 rows of duplicates to look for patterns
pos_count = 0
neg_count = 0
nan_count = 0
unknown = 0
counter= -1
for x in data['total_income']:
    counter +=1
    if x >= 0:
        pos_count +=1
    elif x < 0:
        neg_count +=1
    elif str(x) == 'nan':
        nan_count +=1
        if nan_count < 20:
            print(data.iloc[counter])

    else:
        # to check there are no other values
        unknown +=1
print("Positive income:", pos_count, 
      "\nNegative income:", neg_count,
      "\nnan:\t\t", nan_count,
      "\nUnknown:\t", unknown)


# In[42]:


# categorize income_level into 10K levels

# create a new function, income_level_fx
def income_level_fx(row):
    # the income_level is returned according to total_income
    income = row['total_income']
    
    if income < 10000:
        return 'less_than_10K' 
    elif income < 20000:
        return 'btwn10Kand20K'
    elif income < 30000:
        return 'btwn20Kand30K'   
    elif income < 40000:
        return 'btwn30Kand40K'  
    elif income < 50:
        return 'btwn40Kand50K' 
    elif income < 60000:
        return 'btwn50Kand60K'  
    elif income < 70000:
        return 'btwn60Kand70K' 
    elif income < 80000:
        return 'btwn70Kand80K' 
    elif income < 90000:
        return 'btwn80Kand90K' 
    elif income < 100000:
        return 'btwn90Kand100K'
    elif income >= 100000:
        return 'greater_than_100K'  
    
# create a new column, income_level, based on total_income
data['income_level'] = data.apply(income_level_fx, axis=1)

# verfiy new column: income_level
data['income_level'].value_counts()


# In[43]:


# find info on mean, min, max for total_income
print('Statistical info for total_income')
data['total_income'].describe()


# In[44]:


# find median of total_income
median_income = data['total_income'].median()
print('The median of total_income')
median_income


# total_income is vital to the analysis. Over 10% of the values are missing. 
# Those values could be replaced by the mean or median. 
# 
# Since the mean (26787) > median (23202), high outling values are pulling the mean up. The median should be used to replace the values. A copy of the dataframe, titled data_median_income, will be created to store data where the median total_income replaces the missing values. 
# 
# Another option will be to look at correlations between other factors and total_income and replace missing values based on those factors. 
# 
# The two options will be compared in the final analysis.

# In[45]:


# make a copy of data and replace the missing values with median
data_median_income = data.copy()


# In[46]:


# fill in missing values with median income
data_median_income['total_income'].fillna(value=median_income, inplace = True)


# In[47]:


# print the number of missing values in data_median_income after filling nan
print('Verify that there are no missing values in data_median_income')
data_median_income['total_income'].isnull().sum()


# In[48]:


# categorize data_median_income income_level into 10K levels

# create a new function, income_level_fx
def income_level_fx(row):
    # the income_level is returned according to total_income
    income = row['total_income']
    
    if income < 10000:
        return 'less_than_10K' 
    elif income < 20000:
        return 'btwn10Kand20K'
    elif income < 30000:
        return 'btwn20Kand30K'   
    elif income < 40000:
        return 'btwn30Kand40K'  
    elif income < 50:
        return 'btwn40Kand50K' 
    elif income < 60000:
        return 'btwn50Kand60K'  
    elif income < 70000:
        return 'btwn60Kand70K' 
    elif income < 80000:
        return 'btwn70Kand80K' 
    elif income < 90000:
        return 'btwn80Kand90K' 
    elif income < 100000:
        return 'btwn90Kand100K'
    elif income >= 100000:
        return 'greater_than_100K'  
    
# create a new column, income_level, based on total_income
data_median_income['income_level'] = data_median_income.apply(income_level_fx, axis=1)

# verfiy new column: income_level
data_median_income['income_level'].value_counts()


# In[49]:


# find info on mean, min, max for total_income
print('Statistical info for data_median_income')
data_median_income['total_income'].describe()


# Since debt, children, family_status, purpose will be examined later, those will not be used when considering filling missing values in total_income.
# 
# Before considering relationships between total_income and (gender and/or education_id and/or dob_years and/or income_type), dob_years and education need to be addressed.
# 
# 1. Process dob_years into categories (called age_group) 
# 2. Change the items in education to lowercase so it can be used for analysis.
# 3. Investigate gender, education, dob_years, income_type with total_income

# In[50]:


# process dob_years into age_group

# create a new function, age_group_fx 
def age_group_fx(row):
    # the age group is returned according to dob_years
    age = row['dob_years']

    if age < 20:
        return 'less_than_20' 
    elif age < 30:
        return 'btwn20and30'
    elif age < 40:
        return 'btwn30and40'   
    elif age < 50:
        return 'btwn40and50' 
    elif age < 60:
        return 'btwn50and60' 
    elif age < 70:
        return 'btwn60and70' 
    elif age >= 70:
        return 'over_equal70'  
    
# create a new column, age_group, based on dob_years
data['age_group'] = data.apply(age_group_fx, axis=1)

# calculate number of items per age level
print('Number of items per age group')
data['age_group'].value_counts()


# In[51]:


# calculate number of items per education column
data['education'].value_counts()


# In[52]:


# change items to lowercase
data['education'] = data['education'].str.lower()

# calculate number of items per education column
print('Number of items per education category')
data['education'].value_counts()


# In[53]:


# compare with education #s with education_id #s
print('Number of items per education_id category')
data['education_id'].value_counts()


# The education column and the education_id column values match.

# In[54]:


# investigate gender and total_income
data.groupby('gender').agg({'total_income': ['count', 'mean', 'median']})


# There is close to 5K difference, quite significant, between values for F or M.
# 
# Gender will be used to help fill in missing values for total income.

# In[55]:


# investigate education and total_income
data.groupby('education').agg({'total_income': ['count', 'mean', 'median']})


# There is a significant difference between incomes for different education levels.
# 
# Education will be used to help fill in missing values for total income.

# In[56]:


# investigate count of missing total_income by education
print('Count of missing total income by education')
print(data[data['total_income'].isnull()]['education'].value_counts())


# In[57]:


# investigate age_group and total_income
data.groupby('age_group').agg({'total_income': ['count', 'mean', 'median']})


# There is not as significant difference between incomes for different age groups. 
# 
# Age_groups will not be used to help fill in missing values for total income.

# In[58]:


# investigate income_type and total_income
data.groupby('income_type').agg({'total_income': ['count', 'mean', 'median']})


# There is a sizeable difference between business, employee, and retiree and all of those have over 3K applicants.
# 
# Income_type will be used to help fill in missing values for total income.

# In[59]:


# investigate count of missing total_income by income_type
print('Count of missing total income by income_type')
print(data[data['total_income'].isnull()]['income_type'].value_counts())


# In[60]:


# investigate if there are any correlations with total_income and 
# gender and education and income_type and age group
data.pivot_table(values=['total_income'], columns=['gender', 'education', 'income_type'])


# This suggests some wide variations in value when categorized by gender, education, and income type.

# In[61]:


# fill in values using education_id and income_type
print('Fill in missing values of total_income based on education, gender, and income_type')
data['total_income'] = data['total_income'].fillna(data.groupby(['education_id', 'income_type', 'gender'])['total_income'].transform('median'))

# print the number of missing values in data after filling nan
print('Verify that there are no missing values in data')
data['total_income'].isnull().sum()


# In[62]:


miss_value = data[data['total_income'].isna()]
miss_value


# 1 row still has a missing value for total_income.
# The applicant is a male entrepreneur with a bachelor's degree.

# In[63]:


# investigate if there are any correlations with total_income and 
# gender and income_type
data.pivot_table(values=['total_income'], columns=['gender', 'income_type'])


# Since there is not a listing for a M who is an entrepreneur, the program couldn't fill in a value. The value for a F who is an entrepreneur is very high, so it may be better to base the missing value on a M having a bachelor's degree.

# In[64]:


# investigate if there are any correlations with total_income and 
# gender and education_id
data.pivot_table(values=['total_income'], columns=['gender', 'education'])


# In[65]:


# replace the missing value with value for a M with a bachelor's degree
print('Fill in missing value of total_income based on education and gender')
data['total_income'] = data['total_income'].fillna(data.groupby(['education_id', 'gender'])['total_income'].transform('median'))

# print the number of missing values in data after filling nan
print('Verify that there are no missing values in data')
data['total_income'].isnull().sum()


# In[66]:


# categorize income_level into 10K levels
# important to do after all the missing values added to update

# create a new function, income_level_fx
def income_level_fx(row):
    # the income_level is returned according to total_income
    income = row['total_income']
    
    if income < 10000:
        return 'less_than_10K' 
    elif income < 20000:
        return 'btwn10Kand20K'
    elif income < 30000:
        return 'btwn20Kand30K'   
    elif income < 40000:
        return 'btwn30Kand40K'  
    elif income < 50:
        return 'btwn40Kand50K' 
    elif income < 60000:
        return 'btwn50Kand60K'  
    elif income < 70000:
        return 'btwn60Kand70K' 
    elif income < 80000:
        return 'btwn70Kand80K' 
    elif income < 90000:
        return 'btwn80Kand90K' 
    elif income < 100000:
        return 'btwn90Kand100K'
    elif income >= 100000:
        return 'greater_than_100K'  
    
# create a new column, income_level, based on total_income
data['income_level'] = data.apply(income_level_fx, axis=1)

# verfiy new column: income_level
data['income_level'].value_counts()


# In[67]:


data.info()


# In[68]:


# find info on mean, min, max for total_income
print('Statistical info for total_income')
data['total_income'].describe()


# ### Conclusion

# 1. days_employed may or may not be useful for analysis. Negative values may have occured through human data entry error. Very large values likely occured because of confusion over units (hours versus days), but they might be due to problems merging datasets. To handle the negative values, the absolute value was applied and the new values stored in place. Large values (equivalent to working > 70 years) were divided by 24 (for 24 hours) and replaced. The first 20 rows with NAN values revealed no particular pattern, so the missing values are likely MAR. Then the missing values were filled using the median because there were still large values skewing the results (mean 4641 vs median 2194).
# 
# 
# 2. total_income is vital to analysis (in fact it is one of the key categories to report on). Over 10% of the values are missing. The first 20 duplicates were printed and there is no obvious pattern, therefore these missing values are MAR. Reasons for missing values could include human error, information not provided by applicant, or a mix up when datasets were merged. total_income was categorized into 10K increments and general statistics were displayed. The mean is greater than the median, therefore the median values will be used for replacing missing values.
# 
# 3. Replacing missing values:
# - approach one is to replace those missing values with the median. data_median_income
# - approach two involves filling missing values based on a composite value drawn from the influence of gender, income_type, and education (except for one stray value where only gender and education were used).
# - these approaches will be compared in the final analysis.

# ### Data type replacement

# 1. total_income should be changed from float 64 to an int for visual appeal / ease of understanding.
# 2. Data types could be changed to conserve memory. This is especially useful for very large files.

# In[69]:


# investigate data types
data.info()


# In[70]:


# check memory usage
print('Memory useage before')
data.memory_usage()


# In[71]:


# calculate total memory usage before
print('Memory useage before in MB')
memory_before = data.memory_usage().sum() / (1024**2) #converting to megabytes
memory_before


# In[72]:


# change data types using astype and apply w/numpy
data['children'] = data['children'].astype('int16')
# converting days_employed took 2 steps, 1st numpy to int, then astype
data['days_employed'] = data['days_employed'].apply(np.int)
data['days_employed'] = data['days_employed'].astype('int16')
data['dob_years'] = data['dob_years'].astype('int16')
data['education'] = data['education'].astype('category')
data['education_id'] = data['education_id'].astype('int16')
data['family_status'] = data['family_status'].astype('category')
data['family_status_id'] = data['family_status_id'].astype('int16')
data['gender'] = data['gender'].astype('category')
data['income_type'] = data['income_type'].astype('category')
data['debt'] = data['debt'].astype('int16')
# converting total_income took 2 steps, 1st numpy to int, then astype
data['total_income'] = data['total_income'].apply(np.int)
data['total_income'] = data['total_income'].astype('int16')
data['purpose'] = data['purpose'].astype('category')
data['years_employed'] = data['years_employed'].astype('float32')
data['age_group'] = data['age_group'].astype('category')


# In[73]:


# check memory usage after
print('Memory useage after')
data.memory_usage()


# In[74]:


# calculate total memory usage after
print('Memory useage after in MB')
memory_after = data.memory_usage().sum() / (1024**2) #converting to 
memory_change = memory_before - memory_after
memory_after


# ### Conclusion

# In[75]:


print('Changing data types saved', memory_change, 'MB of memory and changing total_income and days_employed to int type allows for easier reading. Data types were mostly changed using astype, but when changing from float to int apply.np needed to be used. It is good to know that apply.np can only be used on columns with no missing values.') 


# ### Processing duplicates

# 1. Manage duplicate rows
# - Calculate duplicate rows.
# - There is no reason to check for duplicates within columns, as values can repeat.
# 
# 2. Manage purpose column with stemming or lemmatization
# 

# In[76]:


dup_rows = data.duplicated().sum()
pct_duplicated = dup_rows/len(data)
print('There are', dup_rows, 'duplicate rows in the file')
print('Percentage of duplicate rows = {:.2%}'.format(pct_duplicated))


# In[77]:


# remove the duplicate rows
data = data.drop_duplicates()


# In[78]:


# verify that there are no more duplicate rows
print('Number of duplicates after dropping:')
data.duplicated().sum()


# In[79]:


data.info()


# In[80]:


# use stemming to categorize purpose column
from nltk.stem import SnowballStemmer

english_stemmer = SnowballStemmer('english')   

data['purpose_words'] = data['purpose'].str.split().apply(lambda x: [english_stemmer.stem(y) for y in x])

def purpose_group(purpose_words):

    if 'wed' in purpose_words:
        return 'wedding'
    elif 'estat' in purpose_words or 'hous' in purpose_words or 'properti' in purpose_words:
        return 'real_estate'
    elif 'car' in purpose_words:
        return 'car'
    else:# 'educ' or 'uni' in purpose_words:
        return 'education'
    
data['purpose_cat'] = data['purpose_words'].apply(purpose_group)
data.head(10)


# In[81]:


print('Purpose        Count')
data['purpose_cat'].value_counts()


# In[82]:


print('Total number of purpose values')
len(data['purpose_cat'])


# ### Conclusion

# 1. Duplicate rows increased from 54 to 71 after filling in missing values for total_income, but that is still only 0.33%. Duplicate rows can happen when datasets are merged or through human error.
# 2. Duplicate rows were deleted using drop_duplicates() since it is simple to use. Duplicate removal verified.
# 3. 4 categories (real_estate, car, education, wedding) used to filter purpose column.
# 4. Total number of values verified (each purpose_cat assigned a category)

# ### Categorizing Data

# Categorizing data stratifies a large collection of values into groups or levels. It is ideal to use when working with age, income, time or anything that could be continuous but needs to be examined in groups.
# 
# 1. Verify total_income properly categorized income_level for both data df and data_median_income df. See section 1.1 for initial categorizing total_income into income_level.
# 2. Categorize children into child_groups.
# 3. See section 1.1 for categorizing dob_years into age_group

# In[83]:


# verfiy new column: income_level
data['income_level'].value_counts()


# In[84]:


# verfiy new column: income_level
data_median_income['income_level'].value_counts()


# In[85]:


# create new column based on no children, 1 child, 2 or more children
# first investigate problem values
data['children'].value_counts()


# In[86]:


# calculate percentage of problem values (-1)
print('Percent of erroneous values (-1)')
print(47/21454)


# In[87]:


# change the -1 values to 1 
data.loc[data['children'] < 0, ['children']] = 1


# In[88]:


# calculate percentage of problem values (20)
print('Percent of likely erroneous values (20)')
print(76/21454)


# In[89]:


# change the -1 values to 1 
data.loc[data['children'] == 20, ['children']] = 2


# In[90]:


data['children'].value_counts()


# Replaced likely erroneous values (1 for -1) and (2 for 20).

# In[91]:


# create a new function, children_fx
def children_fx(row):
    # the income_level is returned according to total_income
    child = row['children']
    
    if child == 0:
        return 'no children'
    if child == 1:
        return '1 child'
    else:
        return '2 or more children'
    
# create a new column, income_level, based on total_income
data['child_groups'] = data.apply(children_fx, axis=1)
print('New categories for child groups')
data['child_groups'].value_counts()


# In[92]:


# verify child_groups added 
data.head()


# In[93]:


# verify no missing values
print('Total number of entries')
data['child_groups'].count()


# ### Conclusion

# 1. income_level verfied for both data df and data_median_income df. 
# 2. Replaced -1 and 20 values in children, grouped into 3 categories in child_groups. Verified no missing values.

# At this point, clean up df and get rid of cols not used

# In[94]:


# remove days_employed and years_employed columns
del data['days_employed'] 
del data['years_employed']
del data['dob_years']
del data['purpose_words']
del data['education_id']
del data['family_status_id']
data.head()


# ## Answer these questions

# - Is there a relation between having kids and repaying a loan on time?

# In[95]:


# create a formatting rule for ease of visualization
def format_float(value):
    value = value*100
    return f'{value:,.2f}%'
pd.options.display.float_format = format_float


# In[96]:


# create pivot table with percents per child group category
print('Pecentage of applicants with a history of default')
data.pivot_table(values=['debt'], columns=['child_groups'])


# ### Conclusion

# Yes, there is a clear relationship between debt and no children versus debt and any children. Only 7.54% of applicants without a child defaulted, while appliants with 1 or more children defaulted at a rate of 9.17% to 9.29%. Therefore, a applicant with a child may pose a greater default risk.

# - Is there a relation between marital status and repaying a loan on time?

# In[97]:


# create pivot table with percents per marital status category
print('Pecentage of applicants with a history of default')
data.pivot_table(values=['debt'], columns=['family_status'])


# ### Conclusion

# Yes, there are differences in default history amoung applicants with different family status. Widowers pose the best risk, as only 6.57% of them defaulted on a loan in the past. Both married and divorced applicants pose a moderate risk (between 7.11% and 7.55% defaulted in the past). The biggest risk is for unmarried applicants (9.75%) and those in civil partnerships (9.35%). Therefore, appliants who are unmarried or in a civil partnership may pose the greatest default risk.

# - Is there a relation between income level and repaying a loan on time?

# In[98]:


# create pivot table with percents per income level category
print('Pecentage of applicants with a history of default')
print('for per income level where missing values replaced')
print('based on gender, education level, and income type.')
data.pivot_table(values=['debt'], columns=['income_level'])


# In[99]:


# create pivot table with percents per income level category
print('Pecentage of applicants with a history of default')
print('for per income level (where missing values replaced')
print('by median value')
data_median_income.pivot_table(values=['debt'], columns=['income_level'])


# ### Conclusion

# Interestingly, the percentages for both df (the one where 1 median value replaced all missing values and the one where gender, education level, and income type guided replacement values) are very similar.
# 
# Yes, there does seem to be a general trend where the higher the income, the less likely an applicant has defaulted in the past. Those making below60K demonstrate a clear trend towards higher risk as the income drops (7.29%, 7.79%, 8.46%, and then 8.54% for those making less than 20K. The upper half (greater than 60K) defaulted at a rate between 5.13% and 7.23%). Overall, those applicants earning less than 60K may need greater scrutiny as there is a higher likelyhood that they defaulted in the past.

# - How do different loan purposes affect on-time repayment of the loan?

# In[100]:


# create pivot table with percents per purpose category
print('Pecentage of applicants with a history of default')
data.pivot_table(values=['debt'], columns=['purpose_cat'])


# ### Conclusion

# Yes, once again there is a relationship between the purpose of the loan and a history of default. Applicants wishing to purchase a car have the greatest historical defalut rate (9.36%), followed by those who want money for education (9.22%). Applicants looking to buy real estate pose the least risk, as only 7.23% defaulted in the past.
# 

# Investigate other columns for potential relationships to guide future analysis.

# In[101]:


# create pivot table with percents per age group category
print('Pecentage of applicants with a history of default')
data.pivot_table(values=['debt'], columns=['age_group'])


# Age group: Looking into age may be worthwhile as there appears to be a sharp increase in historical default for those under 40.

# In[102]:


# create pivot table with percents per gender category
print('Pecentage of applicants with a history of default')
data.pivot_table(values=['debt'], columns=['gender'])


# Gender: It appears male applicants have a higher rate of default in this sample. Further analysis may be useful.

# In[103]:


# create pivot table with percents per income_type category
print('Pecentage of applicants with a history of default')
data.pivot_table(values=['debt'], columns=['income_type'])


# In[104]:


data['income_type'].value_counts()


# Income type: Only the 4 groups (employee, business, retiree, civil servant) with the greatest number of applicants could be used for analysis since a sample of 1 or 2 isn't useful. Still, there does seem to be a significant difference between the lowest risk (retiree 5.64%) and highest risk (employee 9.57%). It would be worth more analysis.

# ## General conclusion

# Number of children, family status, income, and the stated purpose of the loan can be used to increase the validity of the credit scoring system. 
# 
# Better loan risk (lower percentage of historical defaults on loans):
# - no children
# - widowers
# - married
# - divorced
# - income > 60K
# - purpose of loan: real estate
# - purpose of loan: wedding
# 
# Higher loan risk (higher percentage of historical defaults on loans):
# - have children
# - unmarried
# - in a civil union
# - income < 60K
# - purpose of loan: education
# - purpose of loan: car 
# 
# Additionally, it may be advantageous to pursue analysis of age group, gender, and income type as there do appear to be relationships beween subsections and percentage of defaults.
