
from Bio.ExPASy import Enzyme
import pandas as pd
import re
import os
import wget
import argparse
from src.utils.data import read_pickle, save_to_pickle

def ec_number_to_code(ec_number:str,depth:int=3)->tuple:
    ec_code = [int(i) for i in re.findall('\d+',ec_number.strip())[:depth]]
    return tuple(ec_code + [0]*(depth-len(ec_code)))


def get_ec_class_descriptions(enzclass_path:str)->dict:
    with open(enzclass_path) as handle:
        ec_classes = handle.readlines()[11:-5]

    # Dictionary to store the results
    ec_classes_dict = {}

    # Compile the regex pattern to identify the ID
    pattern = re.compile(r'^(\d+\.\s*(\d+|-)\.\s*(\d+|-)\.-)')

    #Contructs, description based on parents.. not the most efficient but doesn't matter for this case
    def get_deep_label(code):
        level_code = [0,0,0] 
        label = ''
        for level in range(3):
            if code[level]>0:
                level_code[level] = code[level]
                raw_label = ec_classes_dict[tuple(level_code)]['raw_label'].rstrip('.')
                if level>0:
                    raw_label=raw_label[0].lower() + raw_label[1:]
                    prefix = ', '
                else:
                    prefix = ''
                label += prefix + raw_label
        return label

    # Process each line
    for line in ec_classes:
        # Find the ID using the regex
        match = pattern.search(line)
        if match:
            # Extract the ID
            ec_number = match.group(1).strip()
            # Everything after the ID is considered the description
            description = line[match.end():].strip()
            code = ec_number_to_code(ec_number)
            # Add to the dictionary, removing excess spaces and newlines

            ec_classes_dict[code] = {'raw_label':description,'ec_number':ec_number.replace(' ','')}

    # Output the result
    for code in ec_classes_dict.keys():
        ec_classes_dict[code]['label'] = get_deep_label(code)
    
    return ec_classes_dict


def get_ec_number_description(enzyme_dat_path:str,ec_classes:dict)->list:
    with open(enzyme_dat_path) as handle:
        ec_leaf_nodes = Enzyme.parse(handle)
        ec_leaf_nodes = [{'ec_number': record["ID"],'label':record["CA"],'parent_code':ec_number_to_code(record["ID"])} for record in ec_leaf_nodes]

    for leaf_node in ec_leaf_nodes:
        if leaf_node['label']=='':
            leaf_node['label'] = ec_classes[leaf_node['parent_code']]['label']
    return ec_leaf_nodes

def main(output_dir:str):

    ec_classes_data = 'https://ftp.expasy.org/databases/enzyme/enzclass.txt'
    ec_numbers_data = 'https://ftp.expasy.org/databases/enzyme/enzyme.dat'
    enzclass_output_path = os.path.join(output_dir,'enzclass.txt')
    enzyme_dat_output_path = os.path.join(output_dir,'enzyme.dat')
    annotations_output_path = os.path.join(output_dir,'ec_annotations.pkl')

    wget.download(ec_classes_data,out=enzclass_output_path)
    wget.download(ec_numbers_data,out=enzyme_dat_output_path)

    ec_classes = get_ec_class_descriptions(enzclass_output_path)
    ec_numbers = get_ec_number_description(enzyme_dat_output_path,ec_classes)


    ec_annotations = pd.concat([pd.DataFrame.from_records(list(ec_classes.values())),
                                pd.DataFrame.from_records(ec_numbers)],axis=0)[['ec_number','label']]

    ec_annotations['ec_number'] = 'EC:'+ec_annotations['ec_number']
    ec_annotations.set_index('ec_number',inplace=True)

    ec_annotations.index.name=None
    ec_annotations['name']=ec_annotations['synonym_exact'] = ec_annotations['label']
    ec_annotations['synonym_exact'] = ec_annotations['synonym_exact'].apply(lambda x: [x])
    

    save_to_pickle(ec_annotations,annotations_output_path)

if __name__=='__main__':

    parser = argparse.ArgumentParser(
    description="Download OBO file and save GO ID and label to a pickle.")
    parser.add_argument("--output-dir", type=str,required=True,
                        help="Path to save the resulting pickle file.")
    args = parser.parse_args()

    main(args.output_dir)