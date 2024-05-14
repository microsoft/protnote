from src.utils.data import (
    seed_everything,
    log_gpu_memory_usage
)
from src.utils.main_utils import validate_arguments
from src.data.datasets import ProteinDataset, create_multiple_loaders, calculate_sequence_weights
from src.models.ProTCLTrainer import ProTCLTrainer
from src.models.ProTCL import ProTCL
from src.models.protein_encoders import ProteInfer
from src.utils.losses import get_loss
from src.utils.evaluation import EvalMetrics
from src.utils.models import count_parameters_by_layer, sigmoid_bias_from_prob,load_model
from src.utils.configs import get_setup
from src.utils.data import read_json, write_json
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import os
import argparse
import json
from transformers import AutoTokenizer, AutoModel

### SETUP ###
torch.cuda.empty_cache()

def main():
    #---------------------- HANDLE ARGUMENTS ----------------------# 
    parser = argparse.ArgumentParser(
        description="Train and/or Test the ProTCL model.")

    parser.add_argument("--full-path-name", type=str, default=None,
                        help="Specify the desired full path name to define the vocabularies. Defaults to the full path name in the config file.")

    parser.add_argument("--test-paths-names", nargs="+", type=str, default=None,
                        help="Specify all the desired test paths names to test the model using names from config file to test. If not provided, model will not be tested.")
    
    parser.add_argument("--annotations-path-name", type=str, default="GO_ANNOTATIONS_PATH",
                        help="Name of the annotation path. Defaults to GO.")
    
    parser.add_argument("--base-label-embedding-name", type=str, default="GO_BASE_LABEL_EMBEDDING_PATH",
                        help="Name of the base label embedding path. Defaults to GO.")

    parser.add_argument("--load-model", type=str, default=None,
                        help="(Relative) path to the model to be loaded. If not provided, a new model will be initialized.")

    parser.add_argument("--name", type=str, default="ProTCL",
                        help="Name of the W&B run. If not provided, a name will be generated.")

    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="(Relative) path to the configuration file.")

    parser.add_argument("--override", nargs="*",
                        help="Override config parameters in key-value pairs.")
    
    parser.add_argument("--save-prediction-results", action="store_true", default=False,
                        help="Save predictions and ground truth dataframe for validation and/or test")
    
    parser.add_argument("--save-val-test-metrics", action="store_true", default=False,
                        help="Append val/test metrics to json")

    parser.add_argument("--save-val-test-metrics-path", default = 'outputs/results/ablation_results.json',
                        help=" Path to append val/test metrics to json",type=str) 

    args = parser.parse_args()
    validate_arguments(args, parser)


    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8889'

    # Unpack and process the config file
    task = args.annotations_path_name.split('_')[0]
    config = get_setup(
        config_path=args.config,
        run_name=args.name,
        overrides=args.override,
        train_path_name=None,
        val_path_name=None,
        test_paths_names=args.test_paths_names,
        annotations_path_name = args.annotations_path_name,
        base_label_embedding_name = args.base_label_embedding_name,
        amlt=False,
        is_master=False,
    )
    params, paths, timestamp, logger = config["params"], config[
        "paths"], config["timestamp"], config["logger"]

    device = 'cuda:0'
    # Set the GPU device, if using
    torch.cuda.set_device(device)

    # Seed everything so we don't go crazy
    seed_everything(params["SEED"], device)

    # Log the params
    logger.info(json.dumps(params, indent=4))

    # Initialize label tokenizer
    label_tokenizer = AutoTokenizer.from_pretrained(
        params['LABEL_ENCODER_CHECKPOINT'],
    )

    # Initialize label encoder
    label_encoder = AutoModel.from_pretrained(
        params['LABEL_ENCODER_CHECKPOINT'],
    )

    label_encoder = label_encoder.to(device)

    #---------------------- DATASETS ----------------------# 
    # Create individual datasets
    test_dataset = ProteinDataset(
        data_paths=config['dataset_paths']['test'][0], 
        config=config,
        logger=logger,
        require_label_idxs=False,  # Label indices are not required for testing
        label_tokenizer=label_tokenizer,
    ) if args.test_paths_names is not None else None

    # Add datasets to a dictionary
    datasets = {"test": [test_dataset]}
    datasets = {k:v for k,v in datasets.items() if v[0] is not None}
    label_sample_sizes = {"test": None}
    

    logger.info("Initializing data loaders...")
    # Define data loaders
    loaders = create_multiple_loaders(
        datasets,
        params,
        label_sample_sizes=label_sample_sizes,
        shuffle_labels=False,
        in_batch_sampling=False,
        grid_sampler=False,
        num_workers=params["NUM_WORKERS"],
        sequence_weights=None
    )

    # Initialize ProteInfer
    sequence_encoder = ProteInfer.from_pretrained(
        weights_path=paths[f"PROTEINFER_{task}_WEIGHTS_PATH"],
        num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_LABELS"],
        input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
        output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
        kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
        activation=torch.nn.ReLU,
        dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
        num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
        bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
    )

    model = ProTCL(
        # Parameters
        protein_embedding_dim=params["PROTEIN_EMBEDDING_DIM"],
        label_embedding_dim=params["LABEL_EMBEDDING_DIM"],
        latent_dim=params["LATENT_EMBEDDING_DIM"],
        label_embedding_pooling_method=params["LABEL_EMBEDDING_POOLING_METHOD"],
        sequence_embedding_dropout=params["SEQUENCE_EMBEDDING_DROPOUT"],
        label_embedding_dropout=params["LABEL_EMBEDDING_DROPOUT"],
        label_embedding_noising_alpha=params["LABEL_EMBEDDING_NOISING_ALPHA"],

        # Encoders
        label_encoder=label_encoder,
        sequence_encoder=sequence_encoder,
        inference_descriptions_per_label=len(params["INFERENCE_GO_DESCRIPTIONS"].split('+')),

        # Output Layer
        output_mlp_hidden_dim_scale_factor=params["OUTPUT_MLP_HIDDEN_DIM_SCALE_FACTOR"],
        output_mlp_num_layers=params["OUTPUT_MLP_NUM_LAYERS"],
        output_neuron_bias=sigmoid_bias_from_prob(params["OUTPUT_NEURON_PROBABILITY_BIAS"]) if params["OUTPUT_NEURON_PROBABILITY_BIAS"] is not None else None,
        outout_mlp_add_batchnorm=params["OUTPUT_MLP_BATCHNORM"],
        projection_head_num_layers=params["PROJECTION_HEAD_NUM_LAYERS"],
        dropout=params["OUTPUT_MLP_DROPOUT"],
        projection_head_hidden_dim_scale_factor=params["PROJECTION_HEAD_HIDDEN_DIM_SCALE_FACTOR"],

        # Training options
        label_encoder_num_trainable_layers=0,

        # Batch size limits
        label_batch_size_limit=params["LABEL_BATCH_SIZE_LIMIT_NO_GRAD"],
        sequence_batch_size_limit=params["SEQUENCE_BATCH_SIZE_LIMIT_NO_GRAD"],
    ).to(device)


    loss_fn = get_loss(config=config,
                       bce_pos_weight=1,
                       label_weights=None)
    
    # Initialize trainer class to handle model training, validation, and testing
    Trainer = ProTCLTrainer(
        model=model,
        device=device,
        config=config,
        logger=logger,
        timestamp=timestamp,
        run_name=args.name,
        use_wandb=False,
        use_amlt=args.amlt,
        loss_fn=loss_fn,
        is_master=True
    )

    # Log the number of parameters by layer
    count_parameters_by_layer(model.module)

    # Load the model weights if --load-model argument is provided (using the DATA_PATH directory as the root)
    # TODO: Process model loading in the get_setup function
    if args.load_model:
        load_model(
            trainer=Trainer,
            checkpoint_path=os.path.join(config["DATA_PATH"], args.load_model),
            from_checkpoint=args.from_checkpoint
        )
        logger.info(
            f"Loading model checkpoing from {os.path.join(config['DATA_PATH'], args.load_model)}. If training, will continue from epoch {Trainer.epoch+1}.\n")

    # Initialize EvalMetrics
    eval_metrics = EvalMetrics(device=device)
    
    label_sample_sizes = {k:(v if v is not None else len(datasets[k][0].label_vocabulary)) 
                          for k,v in label_sample_sizes.items() if k in datasets.keys()}


    # Setup for testing
    if args.test_paths_names:

        # TODO: If best_val_th is not defined, alert an error to either provide a decision threshold or a validation datapath
        test_metrics = Trainer.evaluate(
            data_loader=loaders["test"][0],
            eval_metrics=eval_metrics.get_metric_collection_with_regex(pattern="f1_m.*",
                                                                        threshold=0.3,
                                                                        num_labels=label_sample_sizes["test"]),
            save_results=args.save_prediction_results,
        )



if __name__ == "__main__":
    main()
