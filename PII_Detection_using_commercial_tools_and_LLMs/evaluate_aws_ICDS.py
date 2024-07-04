import os
import pandas as pd
from nervaluate import Evaluator
import json
import re
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def merge_consecutive_location_gold(df):
    new_rows = []
    prev_row = None

    for index, row in df.iterrows():
        if prev_row is None:
            prev_row = row.copy()
        elif prev_row is not None and prev_row['TYPE']=='LOCATION' and row['TYPE'] == 'LOCATION' and row['start'] - prev_row['end'] in range(1, 4):
            prev_row['text'] += ' ' + row['text']
            prev_row['end'] = row['end']
        else:
            new_rows.append(prev_row)
            prev_row = row.copy() 

    if prev_row is not None:
        new_rows.append(prev_row)

    return pd.DataFrame(new_rows)

def map_aws(dataframe):

    label_mapping = {    'UK_NATIONAL_HEALTH_SERVICE_NUMBER':'ID',
    'DATE_TIME':'DATE',
    'ADDRESS' : 'LOCATION',
    'UK_NATIONAL_INSURANCE_NUMBER':'ID',
    'CA_HEALTH_NUMBER':'ID',
    'IP_ADDRESS':'PHONE',
    'MAC_ADDRESS':'PHONE',
    'AWS_ACCESS_KEY':'ID',
    'CREDIT_DEBIT_CVV':'ID',
    'CREDIT_DEBIT_EXPIRY':'DATE',
    'CREDIT_DEBIT_NUMBER':'ID',
    'DRIVER_ID':'ID',
    'INTERNATIONAL_BANK_ACCOUNT_NUMBER':'ID',
    'LICENSE_PLATE':'ID',
    'PASSWORD':'ID',
    'PIN':'ID',
    'SWIFT_CODE':'ID',
    'VEHICLE_IDENTIFICATION_NUMBER':'ID',      
    'IN_AADHAAR':'ID',
    'IN_PERMANENT_ACCOUNT_NUMBER':'ID',
    'EMAIL':'PHONE',
    'PHONE_OR_FAX':'PHONE',
    'IN_VOTER_NUMBER':'ID',
    'UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER':'ID',
    'US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER':'ID',
    'BANK_ACCOUNT_NUMBER':'ID',
    'BANK_ROUTING':'ID',
    'PASSPORT_NUMBER':'ID',
    'SSN':'ID',
    'IN_NREGA': 'ID',
    'CA_SOCIAL_INSURANCE_NUMBER':'ID',
    'USERNAME':'NAME',
    'URL':'PHONE',
    }      
   
    dataframe['newTYPE'] = dataframe['TYPE'].map(label_mapping).fillna(dataframe['TYPE'])
    del dataframe['TYPE']    
    return dataframe.rename(columns={'newTYPE': 'TYPE'})

def map_gold(dataframe):
    label_mapping = {
    'Country':'LOCATION',
    'Street':'LOCATION',
    'City':'LOCATION',
    'State':'LOCATION',
    'Ward_Location':'LOCATION',
    'Other_Location':'LOCATION',
    'Age':'AGE',
    'Doctor_Name':'NAME',
    'Gaurdian_Name':'NAME',
    'Patient_Name':'NAME',
    'Staff_Name':'NAME',
    'Misc_Medical_ID':'ID', 
    'IP_Address':'PHONE',
    'FAX':'PHONE',
    'Hospital_Name':'LOCATION', 
    'Zip':'LOCATION',
    'Treatment_Date':'DATE',
    'Other_Govt_ID':'ID',
    'Insurance_Number':'ID',
    'Patient_ID':'ID',
    'Landline':'PHONE',
    'Web_url':'PHONE',
    'Device_Number':'ID',
    'Email':'PHONE',
    'Patient_DOB':'DATE',
    'Aadhar':'ID',
    'Driver_License':'ID',
    'PAN_Card':'ID',
    'Voter_ID':'ID',
    'Phone_No':'PHONE'}

    dataframe['newTYPE'] = dataframe['TYPE'].map(label_mapping).fillna(dataframe['TYPE'])
    del dataframe['TYPE']    
    return dataframe.rename(columns={'newTYPE': 'TYPE'})

def remove_prefix(row):
    name_str = row['text']
    start_offset = row['start']

    for prefix in ['Dr. ','Mr. ','Mrs. ']:
        if prefix in name_str:
            row['text'] = name_str.replace(prefix, '')
            row['start'] = start_offset + len(prefix)
    return row

def main():
    root_dir = '/lockbox/confusion_matrix/phi' 
    Final_df = []
    Final_dataframe = pd.DataFrame()

    for subdir, dirs, files in os.walk(root_dir):
        gcp_df = None
        gold_df = None    
    
        for file in files:
            file_path = os.path.join(subdir, file)
            if  'sgpgi' in file:
                gold_df = pd.read_csv(file_path)
    
            if 'aws' in file:
                gcp_df = pd.read_csv(file_path)
    
        if gcp_df is not None and gold_df is not None:

            gcp_df= gcp_df[gcp_df['Score']>0]

            gcp_df = gcp_df.rename(columns={'Type':'TYPE','BeginOffset':'start','EndOffset':'end','Text':'text'})
            gold_df = gold_df.rename(columns={'label':'TYPE','start_offset':'start','end_offset':'end','fake_entity':'text'})
            gold_df = map_gold(gold_df)
            gcp_df = map_aws(gcp_df)
            gold_df.loc[gold_df['TYPE'] == 'NAME'] = gold_df.loc[gold_df['TYPE'] == 'NAME'].apply(remove_prefix, axis=1)
            
            gold_df = merge_consecutive_location_gold(gold_df)
            
            gold_df = gold_df[~gold_df['TYPE'].isin(['Treatment_Time','Gender','Other_Info','Doctor_Specialisation','Patient_Profession'])]
            gcp_df = gcp_df[~gcp_df['TYPE'].isin(['PROFESSION'])]
            
            merged_df = pd.merge(gcp_df, gold_df, on=['start', 'end'], how='outer',suffixes=('_aws', '_gold'))

            Final_df.append( json.loads(merged_df.to_json(orient='records')) )
            Final_dataframe = pd.concat([Final_dataframe, merged_df])

    predicted_labels = Final_dataframe['TYPE_aws'].fillna('OTHERS') 
    true_labels = Final_dataframe['TYPE_gold'].fillna('OTHERS')  

    predicted_labels = predicted_labels.replace('PHONE', 'CONTACT')
    true_labels = true_labels.replace('PHONE', 'CONTACT')
    
    labels = np.unique(true_labels)
    
    true_annotations = [[entity for entity in doc if entity.get('text_gold')] for doc in Final_df]

    for item in true_annotations:
        for entity in item:
            entity['label'] = entity['TYPE_gold']
            
    gcp_annotations  = [[entity for entity in doc if entity.get('text_aws')] for doc in Final_df]
    
    for item in gcp_annotations:
        for entity in item:
            entity['label'] = entity['TYPE_aws']
    
    labels  = ['DATE','NAME','LOCATION','AGE','ID','PHONE']

    results, results_per_tag = Evaluator(true_annotations, gcp_annotations, tags=labels).evaluate()

    with open('/lockbox/confusion_matrix/sgpgi_aws_results.json', 'w') as results_file:
        json.dump(results, results_file, indent=4)
    
    with open('/lockbox/confusion_matrix/sgpgi_aws_results_per_tag.json', 'w') as results_per_tag_file:
        json.dump(results_per_tag, results_per_tag_file, indent=4)
    
    print('Fin!')
           

if __name__ == '__main__':
    main()

