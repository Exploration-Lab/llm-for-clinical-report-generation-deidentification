import os
import json
import google.cloud.dlp
import pandas as pd
import time


PROJECT_ID = 'PROJECT_ID'

FIELDS = ["PERSON_NAME","FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS", "CREDIT_CARD_NUMBER", "ORGANIZATION_NAME", "PHONE_NUMBER","AGE",
          "COUNTRY_DEMOGRAPHIC","DATE","DATE_OF_BIRTH","DOMAIN_NAME","ETHNIC_GROUP","FINANCIAL_ACCOUNT_NUMBER","GENERIC_ID","IBAN_CODE",
          "IP_ADDRESS","LOCATION","LOCATION_COORDINATES","MAC_ADDRESS","MAC_ADDRESS_LOCAL","MEDICAL_RECORD_NUMBER","PASSPORT","STREET_ADDRESS","TIME",
          "URL","VEHICLE_IDENTIFICATION_NUMBER","INDIA_AADHAAR_INDIVIDUAL","INDIA_GST_INDIVIDUAL","INDIA_PAN_INDIVIDUAL"]

def merge_consecutive_names(df):
    new_rows = []
    prev_row = None

    for index, row in df.iterrows():
        if prev_row is None:
            prev_row = row
        elif row['info_type'] == 'PERSON_NAME' and row['start_offset'] == prev_row['end_offset'] + 1:
            prev_row['quote'] += ' ' + row['quote']
            prev_row['end_offset'] = row['end_offset']
        else:
            new_rows.append(prev_row)
            prev_row = row

    if prev_row is not None:
        new_rows.append(prev_row)

    return pd.DataFrame(new_rows)

def resolve_person_name(dataframe):
    for index, row in dataframe[dataframe['info_type'] == 'PERSON_NAME'].iterrows():
        start_offset = row['start_offset']
        end_offset = row['end_offset']
        first_name_exists = any((row['start_offset'] >= start_offset and row['end_offset'] <= end_offset) for index, row in dataframe.iterrows() if row['info_type'] == 'FIRST_NAME')
        last_name_exists =  any((row['start_offset'] >= start_offset and row['end_offset'] <= end_offset) for index, row in dataframe.iterrows() if row['info_type'] == 'LAST_NAME')

        if first_name_exists:
            dataframe.loc[(dataframe['start_offset'] >= start_offset) & (dataframe['end_offset'] <= end_offset) & (dataframe['info_type'] == 'FIRST_NAME'), 'new_info_type'] = 'resolved'

        if last_name_exists:
            dataframe.loc[(dataframe['start_offset'] >= start_offset) & (dataframe['end_offset'] <= end_offset) & (dataframe['info_type'] == 'LAST_NAME'), 'new_info_type'] = 'resolved'

    return dataframe


def map_info_types(dataframe):
    info_type_mapping = {
        "PHONE_NUMBER":"Phone_No",
        "AGE":"Age",
        "DATE":"Treatment_Date",
        "LOCATION":"Location" ,
        "STREET_ADDRESS":"Street",
        "TIME":"Treatment_Time",}

    dataframe['new_info_type'] = dataframe['info_type'].map(info_type_mapping).fillna(dataframe['info_type'])

    return dataframe

def inspect_text_with_dlp(text):
    dlp_client = google.cloud.dlp_v2.DlpServiceClient()

    item = {"value": text}
    info_types = [{"name": field} for field in FIELDS]

    min_likelihood = google.cloud.dlp_v2.Likelihood.LIKELIHOOD_UNSPECIFIED
    max_findings = 0
    include_quote = True

    inspect_config = {
        "info_types": info_types,
        "min_likelihood": min_likelihood,
        "include_quote": include_quote,
        "limits": {"max_findings_per_request": max_findings},
    }

    parent = f"projects/{PROJECT_ID}"

    response = dlp_client.inspect_content(
            request={"parent": parent, "inspect_config": inspect_config, "item": item}
        )

    gcp_df = pd.DataFrame(columns=['quote', 'info_type', 'likelihood','start_offset','end_offset'])

    if response.result.findings:
        for finding in response.result.findings:
            try:
                quote = finding.quote
            except AttributeError:
                quote = None

            info_type = finding.info_type.name
            likelihood = finding.likelihood.name
            end_offset = finding.location.byte_range.end
            start_offset = finding.location.byte_range.start

            gcp_df = gcp_df._append({'quote': quote, 'info_type': info_type, 'likelihood': likelihood, 'start_offset':start_offset,'end_offset':end_offset}, ignore_index=True)

    print('Text inspection via DLP done!!')
    return gcp_df


def adjust_entities(entities, text):
    highlighted_text = ""
    last_end = 0

    for entity in entities:
        start = entity['start_offset']
        end = entity['end_offset']
        label = entity['new_info_type']

        highlighted_text += text[last_end:start]

        adjust_by = 0

        for char in highlighted_text:
          if len(char.encode('utf-8'))>1:
              adjust_by +=  len(char.encode('utf-8')) -1

        entity['start_offset'] -= adjust_by
        entity['end_offset'] -= adjust_by

        last_end = end

def process_jsonl_files(root_dir):

    allowed_folders = set(str(i) for i in range(80, 100))

    for subdir, dirs, files in os.walk(root_dir):
        folder_name = os.path.basename(subdir)
        
        if folder_name in allowed_folders:

            for file in files:
                if file.endswith(".jsonl"):
                    file_path = os.path.join(subdir, file)
    
    
                    with open(file_path, "r") as file:
                        for index, line in enumerate(file, start=1):
                            gcp_folder = os.path.join(subdir.replace('sgpgi_syn_ds','gcp'),f'line_{index}')
                            if not os.path.exists(gcp_folder):
                                os.makedirs(gcp_folder)
    
                            print("Processing file:", file_path, "line:",index)
                            data = json.loads(line.strip())
    
                            text = data.get("text")
                            entities = data.get("entities")
    
                            if entities:
                                sgpgi_df = pd.DataFrame(entities)
                                sgpgi_df =sgpgi_df[['label','start_offset',	'end_offset','fake_entity']]
                                sgpgi_df.to_csv(os.path.join(gcp_folder, 'sgpgi_df.csv'), index=False)
    
                            csv_file_path = os.path.join(gcp_folder, 'gcp_df.csv')
    
                            gcp_df = None
    
                            if not os.path.exists(csv_file_path):
                                try:
                                    gcp_df = inspect_text_with_dlp(text)
                                except Exception as e:
                                    with open('/lockbox/gcp_de_id_script_exceptions.csv','a',newline="") as exc_file:
                                        print('oops check logs!')
                                        writer = csv.writer(exc_file)
                                        writer.writerow([file_path, index, str(e)])
                                    continue
                                    
                                time.sleep(10)
                            if gcp_df is not None:
                                if not gcp_df.empty:
        
                                    gcp_df = map_info_types(gcp_df)
                                    gcp_df = resolve_person_name(gcp_df)
                                    gcp_df =  gcp_df[gcp_df['new_info_type']!='resolved']
                                    gcp_df = merge_consecutive_names(gcp_df)
        
                                    gcp_data = eval(gcp_df.to_json(orient='records'))
                                    adjust_entities(gcp_data, text)
                                    gcp_df = pd.DataFrame(gcp_data)
        
                                    gcp_df = gcp_df[['quote','start_offset', 'end_offset','likelihood','new_info_type']]
                                    gcp_df['quote'] = gcp_df['quote'].apply(lambda x: x.replace('\/','/'))
        
                                    gcp_df.to_csv(csv_file_path, index=False)

def main():

    root_directory = '/lockbox/sgpgi_syn_ds/'
    process_jsonl_files(root_directory)
    
if __name__ == '__main__':
    main()
