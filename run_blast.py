import os
import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore')

from Bio.Blast.Applications import NcbimakeblastdbCommandline, NcbiblastpCommandline
from Bio.Blast import NCBIXML
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from tqdm import tqdm
import time
import argparse
from functools import partial


from src.utils.data import read_fasta, generate_vocabularies
import pandas as pd
import multiprocessing
from src.utils.configs import get_logger
from itertools import product
from joblib import Parallel, delayed, parallel
from tqdm.contrib.concurrent import process_map
import numpy as np
import contextlib

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

class BlastTopHits:
    def __init__(self,
                    db_fasta_path:str,
                    queries_fasta_path:str):
        
        self.db_fasta_path = db_fasta_path
        self.db_path = '.'.join(db_fasta_path.split('.')[:-1])
        self.queries_fasta_path = queries_fasta_path
        
        # test_dict = {uid:go_terms for _,uid,go_terms in test}
        # self.queries = read_fasta(queries_fasta_path)
        self.num_threads = multiprocessing.cpu_count()
        self.logger = get_logger()
        self.db_seq_2_labels = None
        #TODO: Columns of interest should be an argument and a list instead of dict internaly
        # using a mapping from friendly name to actual name in the outfmt
        self.columns = {0:'sequence_name',1:'closest_sequence',2:'percent_seq_identity',3:'e_value',4:'bit_score'}
    
    def make_db(self):    
        makeblastdb_cline = NcbimakeblastdbCommandline(dbtype="prot", input_file=self.db_fasta_path,out=self.db_path)
        _ = makeblastdb_cline()
    
    def blast_db_exists(self):
        required_extensions = ['.pin', '.phr', '.psq']
        for ext in required_extensions:
            if not os.path.exists(f"{self.db_path}{ext}"):
                return False
        return True
    
    def run_blast(self,output_path:str,top_k_hits:int):

        if not self.blast_db_exists():
            self.logger.info("Creating a BLAST database for optimal performance")
            self.make_db()
            self.logger.info("Finished creating database")
        else:
            self.logger.info("Found existing database")
            
        self.logger.info("Initiating BLAST Search")
        get_results_start_time = time.time()
        blastp_cline = NcbiblastpCommandline(query=self.queries_fasta_path, 
                                            # subject=self.db_fasta_path,
                                            db = self.db_path,
                                            outfmt="6 qseqid sseqid pident evalue bitscore",
                                            out=output_path,
                                            num_threads=self.num_threads,
                                            max_target_seqs=top_k_hits
                                            )
        stdout, stderr = blastp_cline()
        get_results_end_time = time.time()
        self.run_duration_seconds = get_results_end_time - get_results_start_time

        results = pd.read_csv(output_path,sep='\t',header=None,names = list(self.columns.values()))
        results = results.loc[results.groupby('sequence_name')['bit_score'].idxmax()]
        results.to_csv(output_path,sep='\t',header=None,index=False)

        self.logger.info(f"BLAST search completed in {self.run_duration_seconds:.2f} seconds.")


    def __parse_blast_line(self,line:str)->dict:
        data = line.strip().split('\t')
        parsed_line = {colname:data[idx] for idx,colname in self.columns.items()}
        return parsed_line
    
    def __transfer_hit_labels(self,parsed_line:dict):
        if self.db_seq_2_labels is None:
            self.db_seq_2_labels = {uid:go_terms for _,uid,go_terms in read_fasta(self.db_fasta_path)}
        parsed_line['transferred_labels'] = self.db_seq_2_labels[parsed_line['closest_sequence']]

    def parse_blast_line(self,line,transfer_labels:bool,flatten_labels:bool = True)->list:
        fully_parsed_line = []
        parsed_line = self.__parse_blast_line(line=line)
        if transfer_labels:
            self.__transfer_hit_labels(parsed_line=parsed_line)
            if flatten_labels:
                #Replicate each parsed_line result once for each label
                transferred_labels = parsed_line.pop('transferred_labels')
                fully_parsed_line.extend([{'transferred_labels':i,**parsed_line} for i in transferred_labels])
            else:
                fully_parsed_line.append(parsed_line)
        else:
            fully_parsed_line.append(parsed_line)

        return fully_parsed_line

    def parse_results(self,blast_results_path:str,transfer_labels:bool,flatten_labels:bool = True)->pd.DataFrame:
        self.logger.info("Parsing BLAST results.")

        
        parse_results_start_time = time.time()
        
        with open(blast_results_path,'r') as handle:                
            lines = handle.readlines()
            wrapper = partial(self.parse_blast_line,
                                transfer_labels=transfer_labels,
                                flatten_labels=flatten_labels)
                
            # parsed_results = process_map(wrapper,handle,max_workers = self.num_threads)
            with tqdm_joblib(tqdm(total=len(lines))) as pbar:
                parsed_results = Parallel(n_jobs=self.num_threads)(delayed(wrapper)(line) for line in lines)

        # Flatten the list of lists into a single list
        flattened_parsed_results = [item for sublist in parsed_results for item in sublist]

        parse_results_end_time = time.time()

        self.parse_results_duration_seconds = parse_results_end_time - parse_results_start_time
        self.logger.info(f"BLAST parsing completed in {self.parse_results_duration_seconds:.2f} seconds.")

        return pd.DataFrame.from_records(flattened_parsed_results)
        

def main():
    parser = argparse.ArgumentParser(description="Run BLAST")
    parser.add_argument("--test-data-path", type=str, required=True, help="The test databse of query sequences")
    parser.add_argument("--train-data-path", type=str, required=False, default='data/swissprot/proteinfer_splits/random/train_GO.fasta', help="The train databse of sequences")
    parser.add_argument("--output_dir", type=str, required=False,default='outputs/results/', help="path to save results")
    parser.add_argument('--top-k-hits',type=int,required=False,default=1,help='The number of top hits to return per query in decreasing hit_expect order')
    parser.add_argument('--max-evalue',type=float,required=False,default=0.05,help='The evalue threshold. Any this with higher evalue than this threshold is omitted from results')
    parser.add_argument("--cache", action="store_true", default=False,help="Whether to cache results if available")

    args = parser.parse_args()
    
    # Suppress all Biopython warnings
    logger = get_logger()
    test = read_fasta(args.test_data_path)
    train = read_fasta(args.train_data_path)
    
    test_name = args.test_data_path.split('/')[-1].split('.')[0]
    train_name = args.train_data_path.split('/')[-1].split('.')[0]
    
    
    raw_results_output_path = os.path.join(args.output_dir,f"blast_raw_{test_name}_{train_name}_results.tsv")
    parsed_results_output_path = os.path.join(args.output_dir,f"blast_parsed_{test_name}_{train_name}_results.parquet")
    pivot_parsed_results_output_path = os.path.join(args.output_dir,f"blast_pivot_parsed_{test_name}_{train_name}_results.parquet")

    bth = BlastTopHits(db_fasta_path=args.train_data_path,
                            queries_fasta_path=args.test_data_path)
    
    if not (os.path.exists(raw_results_output_path) & args.cache):
        bth.run_blast(output_path=raw_results_output_path,top_k_hits=args.top_k_hits)
    
    #Parse and save processed results
    if not (os.path.exists(parsed_results_output_path) & args.cache):
        parsed_results = bth.parse_results(blast_results_path=raw_results_output_path,flatten_labels=False,transfer_labels=True)
        parsed_results.to_parquet(parsed_results_output_path,index=False)
    else:
        parsed_results = pd.read_parquet(parsed_results_output_path)
    
    #Format as pivoted dataframe
    logger.info("Pivoting data")
    db_vocab = generate_vocabularies(file_path = args.train_data_path)['label_vocab']
    label2int = {label:idx for idx,label in enumerate(db_vocab)}

    def record_to_pivot(idx_row):
        _,row = idx_row
        record = [-15.0]*len(db_vocab)
        for l in row['transferred_labels']:
            record[label2int[l]] = 15.0
        record.insert(0,row['sequence_name'])
        return record

 
    #TODO: Could do batch processing for this to avoid holding everything in memory.
    simplified_results = parsed_results[['sequence_name','bit_score','transferred_labels']]

    pivoting_batch_size = 10_000
    num_pivoting_baches = int(np.ceil(len(simplified_results)/10_000))
    simplified_results.iterrows()
    
    for batch in range(num_pivoting_baches):
        logger.info(f"Pivoting batch {batch+1} / {num_pivoting_baches}")
        
        batch_size = min(pivoting_batch_size,len(simplified_results) - (batch)*pivoting_batch_size)

        with tqdm_joblib(tqdm(total=batch_size)) as pbar:
            records = Parallel(n_jobs=multiprocessing.cpu_count()
                               )(delayed(record_to_pivot)(idx_row) 
                                 for idx_row in simplified_results[batch*pivoting_batch_size:(batch+1)*pivoting_batch_size].iterrows())

        result = pd.DataFrame(records,columns = ['sequence_name']+db_vocab)
        result.set_index('sequence_name',inplace=True)
        result.index.name=None
        result.to_parquet(pivot_parsed_results_output_path+f'_batch_{batch}',index=True)

    logger.info(f"Merging batched results.")
    batch_results = []
    for batch in tqdm(range(num_pivoting_baches)):
        batch_results.append(pd.read_parquet(pivot_parsed_results_output_path+f'_batch_{batch}'))
    pd.concat(batch_results).to_parquet(pivot_parsed_results_output_path,index=True)

    logger.info(f"Results saved in {pivot_parsed_results_output_path}")
    logger.info(f'Search Duration: {bth.run_duration_seconds}')
    logger.info(f'Parse Duration: {bth.parse_results_duration_seconds}')
    

if __name__ == "__main__":
    main()