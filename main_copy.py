import psutil
from src.utils.data import (
    load_model_weights,
    seed_everything,
)
from src.utils.main_utils import get_or_generate_vocabularies,  get_or_generate_label_embeddings, get_or_generate_sequence_embeddings, validate_arguments
from src.data.datasets import ProteinDataset, calculate_pos_weight, create_multiple_loaders
from src.models.ProTCLTrainer import ProTCLTrainer
from src.models.ProTCL import ProTCL
from src.models.protein_encoders import ProteInfer
from src.utils.evaluation import EvalMetrics, save_evaluation_results
from src.utils.models import count_parameters_by_layer, sigmoid_bias_from_prob
from src.utils.configs import get_setup
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import wandb
import os
import argparse
import json
from transformers import AutoTokenizer, AutoModel
from src.data.collators import collate_variable_sequence_length

### SETUP ###
torch.cuda.empty_cache()


def print_memory_stats():
    # Print number of threads and cores
    print(f"Number of threads: {psutil.cpu_count()}")
    print(f"Number of physical cores: {psutil.cpu_count(logical=False)}")

    # Get the virtual memory statistics
    memory_info = psutil.virtual_memory()

    # Print total, available, and used RAM
    print(f"Total RAM: {memory_info.total / (1024**3):.2f} GB")
    print(f"Available RAM: {memory_info.available / (1024**3):.2f} GB")
    print(f"Used RAM: {memory_info.used / (1024**3):.2f} GB")

    # Print swap memory details
    swap_info = psutil.swap_memory()
    print(f"Total Swap: {swap_info.total / (1024**3):.2f} GB")
    print(f"Used Swap: {swap_info.used / (1024**3):.2f} GB")
    print(f"Free Swap: {swap_info.free / (1024**3):.2f} GB")

    # Print details about the current process
    process = psutil.Process(os.getpid())
    print(f"Current process ID: {process.pid}")
    print(
        f"Memory usage of current process: {process.memory_info().rss / (1024**3):.2f} GB")
    print(f"Number of open files by the process: {len(process.open_files())}")

    # Print shared memory details
    shm_info = os.statvfs('/dev/shm')
    total_shm = shm_info.f_blocks * shm_info.f_frsize
    free_shm = shm_info.f_bfree * shm_info.f_frsize
    used_shm = total_shm - free_shm
    print(f"Total shared memory (/dev/shm): {total_shm / (1024**3):.2f} GB")
    print(f"Used shared memory (/dev/shm): {used_shm / (1024**3):.2f} GB")
    print(f"Free shared memory (/dev/shm): {free_shm / (1024**3):.2f} GB")

    # Print number of child processes
    print(f"Number of child processes: {len(process.children())}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and/or Test the ProTCL model.")
    parser.add_argument("--train-path-name", type=str, default=None,
                        help="Specify the desired train path name to train the model using names from config file. If not provided, model will not be trained. If provided, must also provide --val-path.")

    parser.add_argument("--validation-path-name", type=str, default=None,
                        help="Specify the desired val path name to validate the model during training using names from config file. If not provided, model will not be trained. If provided, must also provide --train-path.")

    parser.add_argument("--full-path-name", type=str, default=None,
                        help="Specify the desired full path name to define the vocabularies. Defaults to the full path name in the config file.")

    parser.add_argument("--test-paths-names", nargs="+", type=str, default=None,
                        help="Specify all the desired test paths names to test the model using names from config file to test. If not provided, model will not be tested.")

    parser.add_argument("--use-wandb", action="store_true", default=False,
                        help="Use Weights & Biases for logging. Default is False.")

    parser.add_argument("--load-model", type=str, default=None,
                        help="(Relative) path to the model to be loaded. If not provided, a new model will be initialized.")

    parser.add_argument("--name", type=str, default="ProTCL",
                        help="Name of the W&B run. If not provided, a name will be generated.")

    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="(Relative) path to the configuration file.")

    parser.add_argument("--amlt", action="store_true", default=False,
                        help="Run job on Amulet. Default is False.")

    parser.add_argument("--override", nargs="*",
                        help="Override config parameters in key-value pairs.")

    parser.add_argument('-n', '--nodes', default=1, type=int,
                        metavar='N', help='Number of nodes (default: 1)')

    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='Number of gpus per node (default: 1)')

    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='Ranking within the nodes')

    # TODO: This could be more elegant with parser.add_subparsers()
    # TODO: Make Optimization metric and normalize probabilities part of arguments
    args = parser.parse_args()
    validate_arguments(args, parser)

    # TODO: If running with multiple GPUs, make sure the vocabularies and embeddings have been pre-generated (otherwise, it will be generated multiple times)

    # Distributed computing
    args.world_size = args.gpus * args.nodes
    if args.amlt:
        args.nr = int(os.environ['RANK'])
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8889'

    mp.spawn(train_validate_test, nprocs=args.gpus, args=(args,))


def train_validate_test(gpu, args):
    # Calculate GPU rank (based on node rank and GPU rank within the node) and initialize process group
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    print(
        f"### Initializing GPU {gpu+1}/{args.gpus} on node {args.nr}; or, gpu {rank+1}/{args.world_size} for all nodes.")

    # Check if master process
    is_master = rank == 0

    # Unpack and process the config file
    config = get_setup(
        config_path=args.config,
        run_name=args.name,
        overrides=args.override,
        train_path_name=args.train_path_name,
        val_path_name=args.validation_path_name,
        test_paths_names=args.test_paths_names,
        amlt=args.amlt,
        is_master=is_master,
    )
    params, paths, timestamp, logger = config["params"], config[
        "paths"], config["timestamp"], config["logger"]

    # Set the GPU device, if using
    torch.cuda.set_device(gpu)
    device = torch.device('cuda:' + str(gpu)
                          if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize W&B, if using
    if is_master and args.use_wandb:
        wandb.init(
            project="protein-functions",
            name=f"{args.name}_{timestamp}",
            config={**params, **vars(args)},
            entity="microsoft-research-incubation"
        )
        # Log the wandb link
        logger.info(f"W&B link: {wandb.run.get_url()}")

    # Log the params
    logger.info(json.dumps(params, indent=4))

    # Initialize label tokenizer
    label_tokenizer = AutoTokenizer.from_pretrained(
        params['LABEL_ENCODER_CHECKPOINT'])

    # Load or generate the vocabularies
    vocabularies = get_or_generate_vocabularies(
        paths[args.full_path_name], paths["VOCABULARIES_DIR"], logger)

    # Create datasets
    datasets = ProteinDataset.create_multiple_datasets(
        config['dataset_paths_list'],
        label_tokenizer=label_tokenizer,
        vocabularies=vocabularies,
        subset_fractions={
            "train": params["TRAIN_SUBSET_FRACTION"],
            "validation": params["VALIDATION_SUBSET_FRACTION"],
            "test": params["TEST_SUBSET_FRACTION"]}
    )

    # Seed everything so we don't go crazy
    seed_everything(params["SEED"], device)

    # Initialize new run
    logger.info(
        f"################## {timestamp} RUNNING main.py ##################")

    # Define label sample sizes for train, validation, and test loaders
    label_sample_sizes = {
        "train": params["TRAIN_LABEL_SAMPLE_SIZE"],
        "validation": params["VALIDATION_LABEL_SAMPLE_SIZE"],
        "test": None  # No sampling for the test set
    }

    # Define data loaders
    loaders = create_multiple_loaders(
        datasets,
        params,
        label_sample_sizes=label_sample_sizes,
        num_workers=params["NUM_WORKERS"],
        world_size=args.world_size,
        rank=rank,
    )

    # Mappings. A bit hacky, but it works
    # TODO: Is there a cleaner way of doing this? It feels awkward that we define "label2int" based on only train (even though it's the same)
    # Maybe we load label2int from the JSON vocabulary upfront and pass that to the dataset?
    
    label2int = datasets[list(datasets.keys())[0]][0].label2int
    int2label = datasets[list(datasets.keys())[0]][0].int2label
    label_annotation_map = datasets[list(datasets.keys())[
        0]][0].label_annotation_map

    # Load label encoder
    label_encoder = AutoModel.from_pretrained(
        params['LABEL_ENCODER_CHECKPOINT'])


    # Initialize ProteInfer
    sequence_encoder = ProteInfer.from_pretrained(
        weights_path=paths["PROTEINFER_WEIGHTS_PATH"],
        num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_LABELS"],
        input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
        output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
        kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
        activation=torch.nn.ReLU,
        dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
        num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
        bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
    )

    # Generate all sequence embeddings upfront, if not training the sequence encoder
    sequence_embedding_dict = None
    if not params["TRAIN_SEQUENCE_ENCODER"]:
        sequence_embedding_dict = get_or_generate_sequence_embeddings(
            paths,
            device,
            sequence_encoder,
            datasets,
            params,
            logger,
        )
        
    # Generate all label embeddings upfront, if not training the label encoder
    label_embedding_matrix = None
    if not params["TRAIN_LABEL_ENCODER"]:
        # Create a list of text labels
        sorted_labels = sorted(
            vocabularies["GO_label_vocab"], key=lambda x: label2int[x])
        label_annotations = [label_annotation_map[label_id]
                             for label_id in sorted_labels]
        label_encoder = label_encoder.to(device)
        label_embedding_matrix = get_or_generate_label_embeddings(
            paths,
            device,
            label_annotations,
            label_tokenizer,
            label_encoder,
            logger,
            label_batch_size_limit=params["LABEL_BATCH_SIZE_LIMIT"]
        )
        # Move the label encoder to CPU
        label_encoder = label_encoder.cpu()

    print('abc')
    # Loop through all the datasets and set the label and sequence embedding matrices

    for dataset in datasets.values():
        for subset in dataset:
            if not params["TRAIN_LABEL_ENCODER"]:
                subset.set_label_embedding_matrix(label_embedding_matrix.cpu())


    def to_device(device, *args):
        return [item.to(device) if item is not None else None for item in args]

    ####### TRAINING AND VALIDATION LOOPS #######
    if args.train_path_name is not None:
        print("start training")

        for epoch in range(5):
            print("epoch",epoch)
            ####### TRAINING LOOP #######
            for batch in loaders["train"][0]:
                print(batch)
                # Increment batch index
                batch_count += 1
                print(123)

                # Unpack the training batch
                sequence_onehots, sequence_embeddings, sequence_lengths, label_multihots, tokenized_labels, label_embeddings = (
                    batch["sequence_onehots"],
                    batch["sequence_embeddings"],
                    batch["sequence_lengths"],
                    batch["label_multihots"],
                    batch["tokenized_labels"],
                    batch["label_embeddings"]
                )

                # Move all unpacked batch elements to GPU, if available
                sequence_onehots, sequence_embeddings, sequence_lengths, label_multihots, tokenized_labels, label_embeddings = to_device(
                    sequence_onehots, sequence_embeddings, sequence_lengths, label_multihots, tokenized_labels, label_embeddings)
    else:
        logger.info("Skipping training...")
    
    wandb.finish()

    dist.destroy_process_group()
    logger.info("################## train.py COMPLETE ##################")


"""
 Sample usage: python main.py --train-path-name TRAIN_DATA_PATH --validation-path-name VAL_DATA_PATH --test-paths-names TEST_DATA_PATH TEST_DATA_PATH 
 here we pass the same test set twice as an example.
"""
if __name__ == "__main__":
    main()
