import os
import argparse
import pandas as pd
from openai import OpenAI
import json

client = OpenAI(max_retries=5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-tag', type=str)
    parser.add_argument('--output-directory', type=str, default="./datasets/gpt-4_generations/")
    parser.add_argument('--api-file-directory', type=str, default="./api_files/expansion")
    parser.add_argument('--model', type=str, default="gpt-4.1-2025-04-14")
    args = parser.parse_args()
    print(f"Arguments: {args}")
    retrieve_data(args)

def write_jsonl_file(dictlist, path):
    with open(path, 'w') as jsonl_file:
        for d in dictlist:
            jsonl_file.write(json.dumps(d) + '\n')

def retrieve_data(args):
    print(args)
    results_tag = args.results_tag
    output_directory = args.output_directory
    api_file_directory = args.api_file_directory
    model = args.model
    #
    batch_info_filepath = os.path.join(api_file_directory, f"{results_tag}_batch_info.json")
    with open(batch_info_filepath, 'r') as file:
        batch_info = json.load(file)
    batch_id = batch_info['id']
    print("Retrieving batch info!")
    batch_status_info = client.batches.retrieve(batch_id)
    print("Retrieved!")
    print(batch_status_info)
    if batch_status_info.status == "failed":
        print("Batch failed!")
    elif batch_status_info.status == "in_progress":
        print("Batch still being processed!")
    elif batch_status_info.status != "completed":
        print("Misc. Error!")
    else:
        print("Preprocessing retrieved data...")
        output_file_id = batch_status_info.output_file_id
        raw_outputs = client.files.content(output_file_id)
        raw_json_list = raw_outputs.text.split('\n')
        data_list = []
        for raw_json in raw_json_list:
            if len(raw_json.strip()) > 1:
                try:
                    data = json.loads(raw_json.strip())
                    data_list.append(data)
                except:
                    print("Warning: possible issue with the json string. See raw json string below:")
                    print(raw_json)
        #
        print("Done preprocessing, writing raw data to file!")
        raw_data_filepath = os.path.join(output_directory, f"{results_tag}-{model}_raw.jsonl")
        write_jsonl_file(data_list, raw_data_filepath)
        print("Processing raw data...")
        json_list_strings = [x['response']['body']['choices'][0]['message']['content'] for x in data_list]
        master_dict_list = []
        for json_list_string in json_list_strings:
            if len(json_list_string.strip()) > 1:
                try:
                    data = json.loads(json_list_string.strip())
                    for datagroup in data.values():
                        for datapoint in datagroup:
                            master_dict_list.append(datapoint)
                except:
                    print("Warning: possible issue with the json string. See raw json string below:")
                    print(json_list_string)
        print("Writing data to file...")
        data_filepath = os.path.join(output_directory, f"{results_tag}-{model}.jsonl")
        write_jsonl_file(master_dict_list, data_filepath)
        print("All done!")


if __name__=="__main__":
    main()
