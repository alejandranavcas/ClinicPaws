# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:42:49 2023

@author: Alejandra
"""

import warnings
# Temporarily suppress all warnings
warnings.filterwarnings("ignore")

import pandas as pd


# Read info about pacient #####################################################################################################
name_patient = 'Max'
phone_number = '78267819'
pacient_ml_data = pd.read_excel('patient_data/'+name_patient+'_'+phone_number+'/patient_ml_data.xlsx', engine='openpyxl')
patient_info = pd.read_excel('patient_data/'+name_patient+'_'+phone_number+'/patient_info.xlsx', engine='openpyxl')
breed = patient_info.loc[0,'Breed']
age = 2023 - patient_info.loc[0, 'Year of Birth']


# Create dataset to process ##################################################################################################
data = pd.concat([pacient_ml_data, patient_info['Breed']], axis=1)
data['Age'] = [age]

##############################################################################################################################
# Preprocessing

# Where the nan are
nan_columns = data.columns[data.isna().any()].tolist()
no_nan_columns = list(set(data.columns) - set(nan_columns) - set(['Age']))

for c in no_nan_columns:
    # Split the comma-separated categories and use get_dummies
    split_categories = data[c].str.split(', ')
    dummies = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    # add the new dummy variables and delete the categorical 
    result = pd.concat([data, dummies], axis=1)
    result = result.drop(columns=c)

    
if 'Breed' in nan_columns:
    print('Please, enter the breed of the dog')
else:
    # Do one-hot encoding of breed
    result = pd.get_dummies(result, columns=['Breed'])

if 'Ultrasound' in no_nan_columns:
    # Do one-hot encoding of Ultrasound (((and skin lesion)))
    result = pd.get_dummies(result, columns=['Ultrasound'])
    

# Replace NaN values with 0
for col in list(result.columns):
    result[col].fillna(0, inplace=True)

# Resultant preprocessed dataframe
patient_data_pp = result
    
#print("Data of patient preprocessed and ready for ML model")
