import os
import logging
from typing import Literal
from torchdata.datapipes.iter import FileLister, FileOpener
import argparse
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from tqdm import tqdm

def process_sequence_tfrecord(record: dict, annotation_types: list):
    
    sequence = record['sequence'][0].decode()
    id = record['id'][0].decode()

    
    labels = set()
    if 'label' not in record:
        return id,(sequence,labels)
    
    for l in record['label']:
        label = l.decode()
        label_type = label.split(':')[0]
        
        if (label_type in annotation_types):
            labels.add(label)
    return id,(sequence,list(labels))

def process_split(input_path:str,
                  output_path:str,
                  annotation_types: list,
                  split: Literal['train','dev','test']):
    
    datapipe1 = FileLister(input_path, f"{split}*.tfrecord")
    datapipe2 = FileOpener(datapipe1, mode="b")
    tfrecord_loader_dp = datapipe2.load_from_tfrecord()

    records = []
    for idx,record in tqdm(enumerate(tfrecord_loader_dp)):
        id,(sequence,labels) = process_sequence_tfrecord(record,annotation_types)

        description = " ".join(labels)
        record = SeqRecord(Seq(sequence), id=f"{id}", description=description)
        records.append(record)
    
    
    with open(os.path.join(output_path,f"{split}_{'_'.join(annotation_types)}.fasta"), "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-types',nargs = '+', required=True)
    parser.add_argument('--splits',nargs = '+', required=True)
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    input_path = os.path.join(dirname, 'data/swissprot/proteinfer_splits/random/')
    output_path = os.path.join(dirname, 'data/swissprot/proteinfer_splits/random/')

    for split in args.splits:
        logging.info(f'Processing {split} split')
        process_split(input_path = input_path,
                      output_path = output_path,
                      annotation_types=args.annotation_types,
                      split=split)
    

