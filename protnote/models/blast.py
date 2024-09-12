import os
import warnings
warnings.simplefilter("ignore")
from Bio.Blast.Applications import NcbimakeblastdbCommandline, NcbiblastpCommandline
from tqdm import tqdm
import time
from functools import partial
from protnote.utils.data import read_fasta
from protnote.utils.data import tqdm_joblib
from protnote.utils.configs import get_logger
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed


class BlastTopHits:
    def __init__(self, db_fasta_path: str, queries_fasta_path: str):
        self.db_fasta_path = db_fasta_path
        self.db_path = ".".join(db_fasta_path.split(".")[:-1])
        self.queries_fasta_path = queries_fasta_path

        # test_dict = {uid:go_terms for _,uid,go_terms in test}
        # self.queries = read_fasta(queries_fasta_path)
        self.num_threads = multiprocessing.cpu_count()
        self.logger = get_logger()
        self.db_seq_2_labels = None
        # TODO: Columns of interest should be an argument and a list instead of dict internaly
        # using a mapping from friendly name to actual name in the outfmt
        self.columns = {
            0: "sequence_name",
            1: "closest_sequence",
            2: "percent_seq_identity",
            3: "e_value",
            4: "bit_score",
        }

    def make_db(self):
        makeblastdb_cline = NcbimakeblastdbCommandline(
            dbtype="prot", input_file=self.db_fasta_path, out=self.db_path
        )
        _ = makeblastdb_cline()

    def blast_db_exists(self):
        required_extensions = [".pin", ".phr", ".psq"]
        for ext in required_extensions:
            if not os.path.exists(f"{self.db_path}{ext}"):
                return False
        return True

    def run_blast(self, output_path: str, top_k_hits: int):
        if not self.blast_db_exists():
            self.logger.info("Creating a BLAST database for optimal performance")
            self.make_db()
            self.logger.info("Finished creating database")
        else:
            self.logger.info("Found existing database")

        self.logger.info("Initiating BLAST Search")
        get_results_start_time = time.time()
        blastp_cline = NcbiblastpCommandline(
            query=self.queries_fasta_path,
            # subject=self.db_fasta_path,
            db=self.db_path,
            outfmt="6 qseqid sseqid pident evalue bitscore",
            out=output_path,
            num_threads=self.num_threads,
            max_target_seqs=top_k_hits,
        )
        stdout, stderr = blastp_cline()
        get_results_end_time = time.time()
        self.run_duration_seconds = get_results_end_time - get_results_start_time

        results = pd.read_csv(
            output_path, sep="\t", header=None, names=list(self.columns.values())
        )
        results = results.loc[results.groupby("sequence_name")["bit_score"].idxmax()]
        results.to_csv(output_path, sep="\t", header=None, index=False)

        self.logger.info(
            f"BLAST search completed in {self.run_duration_seconds:.2f} seconds."
        )

    def __parse_blast_line(self, line: str) -> dict:
        data = line.strip().split("\t")
        parsed_line = {colname: data[idx] for idx, colname in self.columns.items()}
        return parsed_line

    def __transfer_hit_labels(self, parsed_line: dict):
        if self.db_seq_2_labels is None:
            self.db_seq_2_labels = {
                uid: go_terms for _, uid, go_terms in read_fasta(self.db_fasta_path)
            }
        parsed_line["transferred_labels"] = self.db_seq_2_labels[
            parsed_line["closest_sequence"]
        ]

    def parse_blast_line(
        self, line, transfer_labels: bool, flatten_labels: bool = True
    ) -> list:
        fully_parsed_line = []
        parsed_line = self.__parse_blast_line(line=line)
        if transfer_labels:
            self.__transfer_hit_labels(parsed_line=parsed_line)
            if flatten_labels:
                # Replicate each parsed_line result once for each label
                transferred_labels = parsed_line.pop("transferred_labels")
                fully_parsed_line.extend(
                    [
                        {"transferred_labels": i, **parsed_line}
                        for i in transferred_labels
                    ]
                )
            else:
                fully_parsed_line.append(parsed_line)
        else:
            fully_parsed_line.append(parsed_line)

        return fully_parsed_line

    def parse_results(
        self,
        blast_results_path: str,
        transfer_labels: bool,
        flatten_labels: bool = True,
    ) -> pd.DataFrame:
        self.logger.info("Parsing BLAST results.")

        parse_results_start_time = time.time()

        with open(blast_results_path, "r") as handle:
            lines = handle.readlines()
            wrapper = partial(
                self.parse_blast_line,
                transfer_labels=transfer_labels,
                flatten_labels=flatten_labels,
            )

            # parsed_results = process_map(wrapper,handle,max_workers = self.num_threads)
            with tqdm_joblib(tqdm(total=len(lines))) as pbar:
                parsed_results = Parallel(n_jobs=self.num_threads)(
                    delayed(wrapper)(line) for line in lines
                )

        # Flatten the list of lists into a single list
        flattened_parsed_results = [
            item for sublist in parsed_results for item in sublist
        ]

        parse_results_end_time = time.time()

        self.parse_results_duration_seconds = (
            parse_results_end_time - parse_results_start_time
        )
        self.logger.info(
            f"BLAST parsing completed in {self.parse_results_duration_seconds:.2f} seconds."
        )

        return pd.DataFrame.from_records(flattened_parsed_results)