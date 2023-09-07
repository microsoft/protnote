import os
import logging
from typing import Literal
from torchdata.datapipes.iter import FileLister, FileOpener
import argparse
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm
from src.utils.data import read_fasta

def process_sequence_tfrecord(record: dict, annotation_types: list):
    
    sequence = record['sequence'][0].decode()
    id = record['id'][0].decode()

    
    labels = set()

    #Some rows have no lavel column
    if 'label' not in record:
        return None
    
    #Add all labels from desired annotation types
    for l in record['label']:
        label = l.decode()
        label_type = label.split(':')[0]
        
        if (label_type in annotation_types):
            labels.add(label)

    #Sequence with no annotation from selected types
    if not labels:
        return None

    return id,(sequence,list(labels))

def process_tfrecords(input_dir:str,
                      output_dir:str,
                      annotation_types: list,
                      pattern: str,
                      pattern_name: str
    ):  
    #Load all tfrecords from desired data split
    datapipe1 = FileLister(input_dir, pattern)
    datapipe2 = FileOpener(datapipe1, mode="b")
    tfrecord_loader_dp = datapipe2.load_from_tfrecord()

    records = []
    #Iterate over records, process and write to a fasta file
    for _,record in tqdm(enumerate(tfrecord_loader_dp)):
        processed_sequence = process_sequence_tfrecord(record,annotation_types)

        #Skipping sequence with no labels from desired annotations
        if processed_sequence is None:
            continue
        
        id,(sequence,labels) = processed_sequence

        description = " ".join(labels)
        record = SeqRecord(Seq(sequence), id=f"{id}", description=description)
        records.append(record)
    
    with open(os.path.join(output_dir,f"{pattern_name}_{'_'.join(annotation_types)}.fasta"), "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)
    parser = argparse.ArgumentParser()

    #TODO: I/O paths could be not required and default to some env var
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--annotation-types',nargs = '+', required=True)
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    #TODO: This paths should be in config or something
    #input_dir = os.path.join(dirname, 'data/swissprot/proteinfer_splits/random/')
    #output_dir = os.path.join(dirname, 'data/swissprot/proteinfer_splits/random/')

    patterns = {'train':'train*.tfrecord',
                'dev':'dev*.tfrecord',
                'test':'test*.tfrecord',
                'full':'*.tfrecord'}
    
    for pattern_name,pattern in patterns.items():
        logging.info(f'Processing {pattern_name}')
        process_tfrecords(input_dir = args.input_dir,
                      output_dir = args.output_dir,
                      annotation_types=args.annotation_types,
                      pattern=pattern,
                      pattern_name = pattern_name)
    
    
