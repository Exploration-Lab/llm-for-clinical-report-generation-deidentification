import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import json
import datetime
import logging
import json
import xml.etree.ElementTree as ET

def make_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created at {directory_path}")
    else:
        print(f"Directory already exists at {directory_path}")
        
def summary_to_xml(json_data):

    text_element = ''

    text = json_data["text"]
    ner_entities = json_data["entities"]

    chunks = []
    start_index = 0
    for entity in ner_entities:
        end_index = text.find(entity["fake_entity"], start_index)
        if end_index != -1:
            chunks.append(text[start_index:end_index])
            chunks.append(entity)
            start_index = end_index + len(entity["fake_entity"])
    chunks.append(text[start_index:])

    template = '<PHI TYPE="{}">{}</PHI>'

    for chunk in chunks:
        if isinstance(chunk, dict):
            text_element += template.format(chunk["label"], chunk["fake_entity"])
        else:
            text_element += chunk

    return text_element

def main():
    path_to_folders = '/home/lokesh/synthetic_ds/sgpgi/'
    path_to_summaries = '/lockbox/llama3_20240511/'
    
    sgpgi_ds = []
    folder_path_name = []

    # Iterate through each folder in the specified path
    for folder in os.listdir(path_to_folders):
        folder_path = os.path.join(path_to_folders, folder)
        
        # Check if the item in the folder is a directory
        if os.path.isdir(folder_path):
            folder_path_name.append(folder_path)
            # Iterate through each file in the directory
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # Check if the file is a JSONL file
                if file_name.endswith('.jsonl'):
                    # Open the JSONL file and read each line
                    with open(file_path, 'r') as json_file:
                        text = f'<RECORD ID="{folder}">'
                        for line in json_file:
                            data = json.loads(line)
                            text += summary_to_xml(data)      
                        
                        text += '</RECORD>'
                        
                    sgpgi_ds.append(text)


    model_name = 'llama3'
    make_directory_if_not_exists(path_to_summaries + model_name)
    
    # Configure logging to write to a file
    logging.basicConfig(filename=f'{path_to_summaries}{model_name}_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    tokenizer = AutoTokenizer.from_pretrained("/lockbox/models/Meta-Llama-3-8B-Instruct/",device_map="cuda")
    model = AutoModelForCausalLM.from_pretrained("/lockbox/models/Meta-Llama-3-8B-Instruct/",device_map="cuda")
    
    for summary_number in range(len(folder_path_name)):

        prompt = f"""Generate an extensive discharge summary of at least 2048 words tailored for Indian patients. To ensure authenticity, the generated summary must include distinct patient-specific details like name, age, address, contact information, hospital name, doctor name, and unique ID. Maintain coherence across all the elements, doctor's name, patient’s identity, medications, diseases, etc. Ensure all the PII (personal identifiable information) elements are properly annotated to maintain privacy and authenticity.
                     The generated discharge summary should be XML-formatted with PII annotations. Generated summaries should include following sections: Admission Details, Diagnosis / Chief Complaints, Allergies, Physical Examination, Medical History, Family Medical history, Treatment Plan, Investigations, Medications (List of medications prescribed at discharge), Follow-up Instructions, Procedures/Lab Tests Conducted (List of procedures or tests conducted during hospital stay, along with results if available), and Special Instructions. 
                     Please ensure that these sections are incorporated into the generated summaries, but refrain from including them as tags in the output. The generated summary should be properly enclosed within the <RECORD> and </RECORD> tags to ensure it's within the XML format.

                     Here's an example patient summary:
                     Patient Summary: {sgpgi_ds[summary_number]}."""


        messages = [
            {"role": "user", "content": prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        num_outputs = 15
        outputs = []
        for i in range(num_outputs):
            try:
                output = model.generate(
                    input_ids,
                    max_new_tokens=3000,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9
                )
                response = output[0][input_ids.shape[-1]:]
                generated_summary = tokenizer.decode(response, skip_special_tokens=True)
                file_name = f"llama3_generated_ds{folder_path_name[summary_number].split('/')[-1]}_{i}.txt"
                file_path = f"{path_to_summaries}{model_name}/{file_name}"

                with open(file_path, "w") as file:
                    file.write(generated_summary)

                logging.info(f"Content saved to {file_path}")

                outputs.append(generated_summary)
                
            except (ValueError, Exception) as e:
                logging.error(f"File No.: {folder_path_name[summary_number]} Error occurred: {e}. Moving on to the next iteration...")
                continue

if __name__ == '__main__':
    main()
