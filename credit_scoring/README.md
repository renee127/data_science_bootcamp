# classifying_churn

## Description

This data analysis project uses machine learning and data balancing techniques to create a predictive model higher than the target AUC-ROC (0.93 versus goal of > 0.88).

__Project Overview__
- Interconnect provides landline communication, internet, and several complimentary services. 
- Clients select a monthly payment or yearly contracts.
- Interconnect wants a model for *predicting the churn of clientele*.
- If a user is forecast to leave, they will be offered promotional codes and special plan options.

__Data Description__
- Data is valid as of February 1, 2020.
- Four data files from different souces are provided.
 1. `contract.csv` - contract information
 2. `personal.csv` - the client's personal data
 3. `internet.csv` - information about Interenet services
 4. `phone.csv` - information about telephone services
- The files are linked by the `customerID` column which contains a unique code assigned to each client. 
- The target feature is the `EndDate` column equal to `No`.

__Project Goal__
- Create a classification model that predicts if a customer will leave soon based on the data files supplied.
- AUC-ROC is the primary metric and needs to be at least 0.81, though greater than or equal to 0.88 is ideal.
- Accuracy will also be reported.

## Dependencies
This project requires Python and the following Python libraries installed:

    NumPy
    Pandas
    matplotlib
    seaborn
    math
    time
    functools
    re
    IPython.display
    sklearn
    catboost
    lightgbm
    xgboost
    random
    sys

You will also need to have software installed to run and execute an iPython Notebook.

Authors

Renee Raven

License

This project is licensed under the MIT License - see the LICENSE file for details

