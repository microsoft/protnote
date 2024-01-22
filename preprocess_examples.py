import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import sys
import json
import warnings

def process_descriptions(description, client):
    """
    Takes a list of protein descriptions and returns a list of processed descriptions
    """
    try:
        # Prepend each description with its input number and append with </END>
        formatted_descriptions = [f"<INPUT_{i}>: {desc}</END>" for i, desc in enumerate(description, start=0)]

        # Join the descriptions with newline characters into a string
        description_text = "\n".join(formatted_descriptions)
        
        messages = [
            {"role": "system", "content": "You are an expert biochemist. You understand and speak scientific language."},
            {
                "role": "user",
                "content":
                    "I'm providing you with five separate GO protein annotation descriptions from SwissProt. Your task is to condense these descriptions in a way that, when embedded with a pretrained language model, they will create a meaningful representation in the embedding latent space. " +
                    "This condensed forms should retain the core essence and meaning of each original description and any unique identifying details. " +
                    "The most important information (e.g., the most unique) should come at the END of the sentence."
                    "You do not need to include the word 'protein,' or unecessary words like 'pivotal,' or 'essential' in your condensed form. " +
                    "I will give you batches of sentences, with each sentence terminated by a delimiter </END>. Return a corresponding batch of condensed forms, with each form started with <OUTPUT_i> and terminated by the delimiter </END>." +
                    "Even if two sentences are similar, you should still return two different condensed forms, focused on any differences." +
                    "\n\nHere is an example:\n\n" +
                    "### Inputs (separated by <\END> delimiters):\n" +
                    "<INPUT_0> Any process that activates or increases the frequency, rate or extent of AIM2 inflammasome complex assembly.</END>\n" +
                    "<INPUT_1> The chemical reactions and pathways resulting in the breakdown of phytosteroids, steroids that differ from animal steroids in having substitutions at C24 and/or a double bond at C22.</END>\n" +
                    "<INPUT_2> A specific global change in the metabolism of a bacterial cell (the downregulation of nucleic acid and protein synthesis, and the simultaneous upregulation of protein degradation and amino acid synthesis) as a result of starvation.</END>\n" +
                    "<INPUT_3> Catalysis of the transfer of a phosphate group, usually from ATP, to a substrate molecule.</END>\n" +
                    "<INPUT_4> The process in which the anatomical structures of embryonic epithelia are generated and organized.</END>\n\n" +
                    "### Response (with corresponding outputs terminated with </END>):\n" +
                    "<OUTPUT_0>Activator or accelerator of AIM2 inflammasome complex assembly </END>" +
                    "<OUTPUT_1>Breakdown pathway executor in plants, involving C24 substitutions and C22 double bond, of Phytosteroid .</END>" +
                    "<OUTPUT_2>Downregulates nucleic acid and protein synthesis, upregulates protein degradation and amino acid synthesis during starvation-induced metabolic shift in bacteria.</END>" +
                    "<OUTPUT_3>Transfer catalyst of phosphate group.</END>" +
                    "<OUTPUT_4>Generation and organization of structures of embryonic epithelia.</END>\n\n" +
                    "### And here are a batch of inputs. Ensure that the number of responses is the same as the number of inputs. There must be the same number of </END> in the input and output.`.\n" +
                    description_text
            }
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
        )
        
        return response.choices[0].message.content

    except Exception as e:
        print("Error processing description:", description)
        print(e)
        return None

def parse_outputs(response_text, labels_to_process):
    # Split the response text using the '</END>' delimiter
    items = response_text.strip().split('</END>')

    # Process each item to extract the text after '<OUTPUT_i>'
    processed_descriptions = []
    for item in items:
        if item.strip():
            start_idx = item.find('>') + 1  # Find the closing '>' of the '<OUTPUT_i>'
            description = item[start_idx:].strip()
            processed_descriptions.append(description)
    
    # Assert that the number of processed descriptions is equal to the number of inputs
    if len(processed_descriptions) != len(labels_to_process):
        print(f"Processed descriptions: {processed_descriptions}")
        print(f"Response text: {response_text}")
        print(f"Labels to process: {labels_to_process}")
        warnings.warn("Mismatch between number of processed descriptions and number of labels to process.")
        processed_descriptions = ["<ERROR>"] * len(labels_to_process)    
        
    return processed_descriptions

def main(start_from_pickle=None):
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")

    client = OpenAI(api_key=api_key)
    batch_size = 1000
    if start_from_pickle:
        df = pd.read_pickle(start_from_pickle)
    else:
        data_file = '/home/ncorley/proteins/ProteinFunctions/data/annotations/go_annotations_2019_07_01.pkl'
        df = pd.read_pickle(data_file)
        df['processed_label'] = None

    # Determine the index to start from
    first_unprocessed_idx = df.index[df['processed_label'].isnull()].tolist()
    start_index = first_unprocessed_idx[0] if first_unprocessed_idx else 0
    start_pos = df.index.get_loc(start_index)

    # Calculate the batch number to start from
    batch_num_start = start_pos // batch_size
    
    # Total number of batches
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)

    for batch_num in tqdm(range(batch_num_start, num_batches), desc="Processing batches"):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        batch_labels = df['label'][start_idx:end_idx]
        print(f"Starting batch {batch_num}, Starting index {start_idx}, Ending index {end_idx}")
        
        # Build processed labels in batches
        processed_labels = []
        micro_batch_size = 5
        for i in range(0, len(batch_labels), micro_batch_size):
            # Select labels for processing, up to the micro batch size
            labels_to_process = batch_labels[i:i+micro_batch_size].tolist()

            # Process the group of labels
            response_text = process_descriptions(labels_to_process, client)
            processed_outputs = parse_outputs(response_text, labels_to_process)

            # Append processed outputs to the list
            processed_labels.extend(processed_outputs[:len(labels_to_process)])
            
            # Print progress every 100 labels
            if i % 100 == 0:
                print(f"Processed {i} labels of batch {batch_num}")

        # Assign the processed labels to the DataFrame
        df.iloc[start_idx:end_idx, df.columns.get_loc('processed_label')] = processed_labels

        # Save the progress after each batch
        df.to_pickle(f'/home/ncorley/proteins/ProteinFunctions/data/annotations/processed_labels/updated_dataframe_2019_augmented_v1_batch_{batch_num}.pkl')

if __name__ == "__main__":
    start_from = None
    if len(sys.argv) > 1:
        start_from = sys.argv[1]
        print(f"Starting from batch {start_from}")
    main(start_from)