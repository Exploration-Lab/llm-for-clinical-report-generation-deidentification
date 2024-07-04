import os
import json
import pandas as pd
import time
import boto3
import math
import csv

os.environ['AWS_ACCESS_KEY_ID'] = "AWS_ACCESS_KEY_ID"
os.environ['AWS_SECRET_ACCESS_KEY'] = "AWS_SECRET_ACCESS_KEY"
os.environ['AWS_REGION'] = "AWS_REGION"

comprehend = boto3.client('comprehendmedical',region_name="AWS_REGION")

def inspect_text_with_comprehend(text):

    chunk_size = 5000
    final_entities = []
    num_chunks = math.ceil(len(text) / chunk_size)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(text))
        limit_text = text[start_idx:end_idx]

        detected_entities = (comprehend.detect_phi(
            Text=limit_text,
        ))

        for x in detected_entities['Entities']:
            x['Text'] = limit_text[x['BeginOffset']:x['EndOffset']]
            x['BeginOffset'] += start_idx
            x['EndOffset'] += start_idx

        final_entities.extend(detected_entities['Entities'])

    return pd.DataFrame(final_entities)

def adjust_entities(entities, text):
    highlighted_text = ""
    last_end = 0

    for entity in entities:
        start = entity['BeginOffset']
        end = entity['EndOffset']
        label = entity['Type']

        highlighted_text += text[last_end:start]

        adjust_by = 0

        for char in highlighted_text:
          if len(char.encode('utf-8'))>1:
              adjust_by +=  len(char.encode('utf-8')) -1

        entity['BeginOffset'] -= adjust_by
        entity['EndOffset'] -= adjust_by

        last_end = end

def process_jsonl_files(root_dir):
    print('starting up..')

    allowed_folders = set(str(i) for i in range(80, 100))

    for subdir, dirs, files in os.walk(root_dir):
        folder_name = os.path.basename(subdir)
        if folder_name in allowed_folders:

            for file in files:
                if file.endswith(".jsonl"):
                    file_path = os.path.join(subdir, file)
    
                    with open(file_path, "r") as file:
                        for index, line in enumerate(file, start=1):
                            #print("Processing file:", file_path, "line:",index)                            
                            aws_folder = os.path.join(subdir.replace('sgpgi_ds','confusion_matrix/phi'),f'line_{index}')
                            if not os.path.exists(aws_folder):
                                os.makedirs(aws_folder)
    
                            data = json.loads(line.strip())
    
                            text = data.get("text")
                            entities = data.get("entities")
    
                            if entities:
                                sgpgi_df = pd.DataFrame(entities)
                                sgpgi_df =sgpgi_df[['label','start_offset',	'end_offset','fake_entity']]
                                sgpgi_df.to_csv(os.path.join(aws_folder, 'sgpgi_df.csv'), index=False)
    
                            aws_df = None
                            csv_file_path = os.path.join(aws_folder, f"aws_df.csv")
    
                            if not os.path.exists(csv_file_path):
                                try:
                                    aws_df = inspect_text_with_comprehend(text)
                                except Exception as e:
                                    with open('/lockbox/aws_de_id_script_exceptions.csv','a',newline="") as exc_file:
                                        print('oops check logs!')
                                        writer = csv.writer(exc_file)
                                        writer.writerow([file_path, index, str(e)])
                                    continue
                                    
                                time.sleep(10)

                            if aws_df is not None:
                                if not aws_df.empty:
        
                                    aws_data = eval(aws_df.to_json(orient='records'))
                                    adjust_entities(aws_data, text)
                                    aws_df = pd.DataFrame(aws_data)
        
                                    aws_df.to_csv(csv_file_path, index=False)

def main():

    root_directory = '/lockbox/sgpgi_ds/'
    process_jsonl_files(root_directory)
    
if __name__ == '__main__':
    main()

