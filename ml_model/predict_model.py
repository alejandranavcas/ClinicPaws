# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 12:49:25 2023

@author: Alejandra
"""

#pip install numpy
#pip install pandas=1.5.3
#pip install xlrd (maybe not needed)
#pip install openpyxl
#pip install scikit-learn=1.2.1


import warnings
# Temporarily suppress all warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle as pkl

#%%
# Upload patient data ###############################################################################################
from preprocessing_data_patient import * # what we are using is patient_data_pp

#%%
# Process by hand ####################################################################################################

# Define the columns you want in your DataFrame
columns = ['Age (Years)', 'Anorexia', 'Depression', 'Fever', 'Lethargy',
       'Weight Loss', 'Cough', 'Cyanosis', 'Dyspnea', 'Pleural Effusion',
       'Respiratory Distress', 'Tachypnea', 'Diarreah',
       'Intestinal Blood Loss', 'Tenesmus', 'Hypercalcemia',
       'Hyperglobulinemia', 'Hypoalbuminemia', 'Mature Neutrophilia',
       'Mild Nonregenerative Anemia', 'Neutrophilia with Left Shift',
       'Nonregenerative Anemia', 'Thrombocytopenia', 'Calcium UP',
       'Hepatic enzymes UP', 'Within Reference Ranges',
       'Peripheral Lymphadenopathy', 'Yes', 'Ataxia',
       'Central Vestibular Disease', 'Cervical Pain',
       'Multifocal Cranial Nerve Involvement', 'Papilledema', 'Seizure',
       'Tetraparesis', 'Granulomatous Chorioretinitis', 'Optic Neuritis',
       'Retinal Hemorrhage', 'Breed_American Cocker Spaniel', 'Breed_Beagle',
       'Breed_Boston Terrier', 'Breed_Boxer', 'Breed_Brittany',
       'Breed_Bulldog', 'Breed_Chihuahua', 'Breed_Cocker Spaniel',
       'Breed_Dachshund', 'Breed_Dalmatian', 'Breed_Doberman',
       'Breed_Doberman Pinscher', 'Breed_German Sheperd',
       'Breed_Golden Retriever', 'Breed_Great Dane',
       'Breed_Labrador Retriever', 'Breed_Mastiff', 'Breed_Pointer',
       'Breed_Pomeranian', 'Breed_Pug', 'Breed_Rottweiler',
       'Breed_Saint Bernard', 'Breed_Shih Tzu', 'Breed_Siberian Husky',
       'Breed_Weimaraners', 'Breed_Wiemaraners', 'Breed_Yorkshire Terrier',
       'Ultrasound_Organomegaly']

# Create an empty DataFrame with the specified columns
df = pd.DataFrame(columns=columns)

# Create a single row with NaN values
nan_values = [np.nan] * len(columns)
df = df.append(pd.Series(nan_values, index=columns), ignore_index=True)


for c in columns:
    try:
        df[c] = int(patient_data_pp[c])
        #print(c, 'is a feature in the dog.')
    except:
        df[c] = 0
        #print(c, 'is not there.')


#%%
# Make a prediction #####################################################################################################

new_pacient = df.to_numpy()
new_pacient = np.reshape(new_pacient, (1, -1))

# Open the saved file with read-binary mode containing the model
model = pkl.load(open('ml_model/model.pickle', 'rb'))

# Use the loaded model to make predictions 
prediction = np.asarray(model.predict(new_pacient))[0]
print('Prediction: ', prediction)

result_from_python = 'Prediction: ' + prediction