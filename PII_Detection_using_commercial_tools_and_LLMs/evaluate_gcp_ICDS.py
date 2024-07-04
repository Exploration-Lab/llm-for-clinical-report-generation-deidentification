import os
import pandas as pd
from nervaluate import Evaluator
import json
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def map_sgpgi(dataframe):

    label_mapping = {
    'Treatment_Date': 'DATE',
    'PERSON_NAME' : 'NAME',
    'Location':'LOCATION',
    'ORGANIZATION_NAME':'LOCATION',
    'IP_ADDRESS':'PHONE',
    'Phone_No':'PHONE',
    'IP_Address': 'PHONE',
    'Patient_DOB': 'DATE',      
    'EMAIL_ADDRESS':'PHONE',
    'CREDIT_CARD_NUMBER':'ID',
    'PASSPORT':'ID',
    'URL':'PHONE',
    "VEHICLE_IDENTIFICATION_NUMBER":'ID',
    "INDIA_AADHAAR_INDIVIDUAL":'ID',
    "INDIA_GST_INDIVIDUAL":'ID',
    "INDIA_PAN_INDIVIDUAL":'ID',
    'DATE_OF_BIRTH':'DATE',
    'DOMAIN_NAME':'PHONE',
    'FINANCIAL_ACCOUNT_NUMBER':'ID',
    'GENERIC_ID':'ID',
    'MEDICAL_RECORD_NUMBER':'ID',
    'IBAN_CODE':'ID',
    'LOCATION_COORDINATES':'LOCATION',
    'MAC_ADDRESS':'PHONE',
    'MAC_ADDRESS_LOCAL':'PHONE',
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
    'FAX':'PHONE',
    'Hospital_Name':'LOCATION', 
    'Zip':'LOCATION',
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

    dataframe['new_gcp_label'] = dataframe['gcp_label'].map(label_mapping).fillna(dataframe['gcp_label'])
    dataframe['new_sgpgi_label'] = dataframe['sgpgi_label'].map(label_mapping).fillna(dataframe['sgpgi_label'])

    del dataframe['gcp_label']    
    del dataframe['sgpgi_label']    

    return dataframe.rename(columns={'new_sgpgi_label': 'sgpgi_label','new_gcp_label':'gcp_label'})

def resolve_person_name_gcp(dataframe):
    for index, row in dataframe.iterrows():
        if row['new_info_type'] == 'PERSON_NAME':
            start_offset = row['start_offset']
            end_offset = row['end_offset']
            first_name_exists = any((row['start_offset'] >= start_offset and row['end_offset'] <= end_offset) for index, row in dataframe.iterrows() if row['new_info_type'] == 'FIRST_NAME')
            last_name_exists =  any((row['start_offset'] >= start_offset and row['end_offset'] <= end_offset) for index, row in dataframe.iterrows() if row['new_info_type'] == 'LAST_NAME')

            if first_name_exists:
                dataframe.loc[(dataframe['start_offset'] >= start_offset) & (dataframe['end_offset'] <= end_offset) & (dataframe['new_info_type'] == 'FIRST_NAME'), 'new_info_type'] = 'resolved'
                
            if last_name_exists:
                dataframe.loc[(dataframe['start_offset'] >= start_offset) & (dataframe['end_offset'] <= end_offset) & (dataframe['new_info_type'] == 'LAST_NAME'), 'new_info_type'] = 'resolved'
    
    return dataframe


def main():
    root_dir = '/lockbox/gcpl/'
    Final_df =  pd.DataFrame()
    Final_list =  []

    allowed_folders = set(str(i) for i in range(80, 100))

    for subdir, dirs, files in os.walk(root_dir):
        gcp_df = None
        sgpgi_df = None    

        subdir_split = subdir.split('/')

        if len(subdir_split) >= 3 and subdir_split[3] in allowed_folders: 
        
            for file in files:

                file_path = os.path.join(subdir, file)
                if file == 'sgpgi_df.csv':
                    sgpgi_df = pd.read_csv(file_path)
        
                if file == 'gcp_df.csv':
                    gcp_df = pd.read_csv(file_path)
        
        if gcp_df is not None and sgpgi_df is not None:
    
            gcp_df = resolve_person_name_gcp(gcp_df)
    
            merged_df = pd.merge(gcp_df, sgpgi_df, on=['start_offset', 'end_offset'], how='outer')
            
            merged_df = merged_df.rename(columns={
                'quote': 'gcp_entity',
                'fake_entity': 'sgpgi_entity',
                'likelihood': 'gcp_likelihood',
                'new_info_type': 'gcp_label',
                'label': 'sgpgi_label',
            })
            
            merged_df = merged_df[~merged_df['sgpgi_label'].isin(['Gender','Treatment_Time','Other_Info','Doctor_Specialisation','Patient_Profession'])]
            merged_df = merged_df[~merged_df['gcp_label'].isin(['Treatment_Time','resolved','Doctor_Specialisation','Patient_Profession'])]
            
            merged_df = map_sgpgi(merged_df)

            Final_df = pd.concat([Final_df, merged_df])

            Final_list.append( json.loads(merged_df.to_json(orient='records')) )

    true_annotations = [[entity for entity in doc if entity.get('sgpgi_entity')] for doc in Final_list]

    for item in true_annotations:
        for entity in item:
            entity['label'] = entity['sgpgi_label']
            entity['start'] = entity['start_offset']
            entity['end'] = entity['end_offset']

    gcp_annotations  = [[entity for entity in doc if entity.get('gcp_entity')] for doc in Final_list]
    for item in gcp_annotations:
        for entity in item:
            entity['label'] = entity['gcp_label']
            entity['start'] = entity['start_offset']
            entity['end'] = entity['end_offset']

    labels = ['DATE','LOCATION','NAME','ID','AGE','PHONE']
    
    results, results_per_tag = Evaluator(true_annotations, gcp_annotations, tags=labels).evaluate()

    with open('gcp_results.json', 'w') as results_file:
        json.dump(results, results_file, indent=4)

    with open('gcp_results_per_tag.json', 'w') as results_per_tag_file:
        json.dump(results_per_tag, results_per_tag_file, indent=4)

    print('Fin!')
           
if __name__ == '__main__':
    main()

