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
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import wandb
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

    parser.add_argument('--from-checkpoint', action="store_true", default=False,
                        help="Continue training from a previous model checkpoint (including optimizer state and epoch). Default is False.")

    parser.add_argument("--name", type=str, default="ProTCL",
                        help="Name of the W&B run. If not provided, a name will be generated.")

    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="(Relative) path to the configuration file.")

    parser.add_argument("--amlt", action="store_true", default=False,
                        help="Run job on Amulet. Default is False.")

    parser.add_argument("--mlflow", action="store_true", default=False,
                        help="Use MLFlow. Default is False.")

    parser.add_argument("--override", nargs="*",
                        help="Override config parameters in key-value pairs.")
    
    parser.add_argument("--save-prediction-results", action="store_true", default=False,
                        help="Save predictions and ground truth dataframe for validation and/or test")
    
    parser.add_argument("--save-val-test-metrics", action="store_true", default=False,
                        help="Append val/test metrics to json")

    parser.add_argument("--save-val-test-metrics-path", default = 'outputs/results/ablation_results.json',
                        help=" Path to append val/test metrics to json",type=str) 
    
    parser.add_argument('-n', '--nodes', default=1, type=int,
                        metavar='N', help='Number of nodes (default: 1)')

    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='Number of gpus per node (default: 1)')

    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='Ranking within the nodes')
    
    args = parser.parse_args()
    validate_arguments(args, parser)

    # TODO: If running with multiple GPUs, make sure the vocabularies and embeddings have been pre-generated (otherwise, it will be generated multiple times)

    # Distributed computing
    args.world_size = args.gpus * args.nodes
    if args.amlt:
        os.environ['MASTER_ADDR'] = os.environ['MASTER_IP']
        args.nr = int(os.environ['NODE_RANK'])
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
    print(f"{'=' * 50}\n"
          f"Initializing GPU {gpu}/{args.gpus-1} on node {args.nr};\n"
          f"    or, gpu {rank+1}/{args.world_size} for all nodes.\n"
          f"{'=' * 50}")

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
    
    # Seed everything so we don't go crazy
    seed_everything(params["SEED"], device)

    # Initialize W&B, if using
    if is_master and args.use_wandb:
        wandb.init(
            project="protein-functions",
            name=f"{args.name}_{timestamp}",
            config={**params, **vars(args)},
            sync_tensorboard=False,
            entity="protein-functions"
        )

        if args.amlt & args.mlflow:
            import mlflow

            #MLFlow logging for Hyperdrive
            mlflow.autolog()
            mlflow.start_run()

        # Log the wandb link
        logger.info(f"W&B link: {wandb.run.get_url()}")

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
    if params["GRADIENT_CHECKPOINTING"]:
        raise NotImplementedError(
            "Gradient checkpointing is not yet implemented.")

    label_encoder = label_encoder.to(device)

    #---------------------- DATASETS ----------------------# 
    # Create individual datasets
    train_dataset = ProteinDataset(
        data_paths=config['dataset_paths']['train'][0], 
        config=config,
        logger=logger,
        require_label_idxs=params['GRID_SAMPLER'],
        label_tokenizer=label_tokenizer,
    ) if args.train_path_name is not None else None

    validation_dataset = ProteinDataset(
        data_paths=config['dataset_paths']['validation'][0], 
        config=config,
        logger=logger,
        require_label_idxs=False,  # Label indices are not required for validation. 
        label_tokenizer=label_tokenizer,
    ) if args.validation_path_name is not None else None

    test_dataset = ProteinDataset(
        data_paths=config['dataset_paths']['test'][0], 
        config=config,
        logger=logger,
        require_label_idxs=False,  # Label indices are not required for testing
        label_tokenizer=label_tokenizer,
    ) if args.test_paths_names is not None else None

    # Add datasets to a dictionary
    # TODO: This does not support multiple datasets. But I think we should remove that support anyway. Too complicated.
    datasets = {
        "train": [train_dataset],
        "validation": [validation_dataset],
        "test": [test_dataset]
    }

    #Remove empty datasets. May happen in cases like only validating a model.
    datasets = {k:v for k,v in datasets.items() if v[0] is not None}


    #-----------------------------------------------------# 

    # Initialize new run
    logger.info(
        f"################## {timestamp} RUNNING main.py ##################")

    # Define label sample sizes for train, validation, and test loaders
    label_sample_sizes = {
        "train": params["TRAIN_LABEL_SAMPLE_SIZE"],
        "validation": params["VALIDATION_LABEL_SAMPLE_SIZE"],
        "test": None,  # No sampling for the test set
    }
    
    # Calculate the weighting for the train dataset
    sequence_weights = None
    if params["WEIGHTED_SAMPLING"] & (args.train_path_name is not None):
        # Calculate label weights
        logger.info("Calculating label weights for weighted sampling...")
        label_weights = datasets["train"][0].calculate_label_weights(power=params["INV_FREQUENCY_POWER"])

        # Calculate sequence weights
        logger.info("Calculating sequence weights based on the label weights...")
        sequence_weights = calculate_sequence_weights(datasets["train"][0].data, label_weights)
        
        # If using clamping, clamp the weights based on the hyperparameters
        if params["SAMPLING_LOWER_CLAMP_BOUND"] is not None:
            sequence_weights = [max(x, params["SAMPLING_LOWER_CLAMP_BOUND"]) for x in sequence_weights]
        if params["SAMPLING_UPPER_CLAMP_BOUND"] is not None:
            sequence_weights = [min(x, params["SAMPLING_UPPER_CLAMP_BOUND"]) for x in sequence_weights]
            
    logger.info("Initializing data loaders...")
    # Define data loaders
    loaders = create_multiple_loaders(
        datasets,
        params,
        label_sample_sizes=label_sample_sizes,
        shuffle_labels=params['SHUFFLE_LABELS'],
        in_batch_sampling=params['IN_BATCH_SAMPLING'],
        grid_sampler=params['GRID_SAMPLER'],
        num_workers=params["NUM_WORKERS"],
        world_size=args.world_size,
        rank=rank,
        sequence_weights=sequence_weights
    )

    if params["LABEL_ENCODER_NUM_TRAINABLE_LAYERS"]==0:
        # Move the label encoder to CPU
        label_encoder = label_encoder.cpu()

    # Initialize ProteInfer
    if params['PRETRAINED_SEQUENCE_ENCODER']: 
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
    else:
        sequence_encoder = ProteInfer(
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
        label_encoder_num_trainable_layers=params["LABEL_ENCODER_NUM_TRAINABLE_LAYERS"],
        train_sequence_encoder=params["TRAIN_SEQUENCE_ENCODER"], 

        # Batch size limits
        label_batch_size_limit=params["LABEL_BATCH_SIZE_LIMIT_NO_GRAD"],
        sequence_batch_size_limit=params["SEQUENCE_BATCH_SIZE_LIMIT_NO_GRAD"],

        #Others
        feature_fusion=config["params"]["FEATURE_FUSION"],
        temperature=config["params"]["SUPCON_TEMP"]
    ).to(device)


    # Wrap the model in DDP for distributed computing
    if config["params"]["SYNC_BN"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model,
                device_ids=[gpu],
                find_unused_parameters=True)

    # Calculate bce_pos_weight based on the training set
    if (params["BCE_POS_WEIGHT"] is None) & (args.train_path_name is not None):
        bce_pos_weight = datasets["train"][0].calculate_pos_weight().to(device)
    elif (params["BCE_POS_WEIGHT"] is not None):
        bce_pos_weight = torch.tensor(params["BCE_POS_WEIGHT"]).to(device)
    else:
        raise ValueError(
            "BCE_POS_WEIGHT is not provided and no training set is provided to calculate it.")

    if (params["LOSS_FN"]=='WeightedBCE'):
        if (args.train_path_name is not None):
            logger.info('calculating WEIGHTED BCE WEIGHTS')
            label_weights = datasets["train"][0].calculate_label_weights(inv_freq=True,
                                                                         normalize = True, 
                                                                         return_list=True,
                                                                         power = params["INV_FREQUENCY_POWER"]).to(device)
        else:
            raise ValueError("Must provde training set")
        
    elif (params["LOSS_FN"]=='CBLoss'):
        if (args.train_path_name is not None):
            label_weights = datasets["train"][0].calculate_label_weights(inv_freq=False,
                                                                         normalize = False,
                                                                         return_list=True,
                                                                         power = params["INV_FREQUENCY_POWER"]).to(device)
        else:
            raise ValueError("Must provde training set")
    else:
        label_weights = None

    loss_fn = get_loss(config=config,
                       bce_pos_weight=bce_pos_weight,
                       label_weights=label_weights)
    
    # Initialize trainer class to handle model training, validation, and testing
    Trainer = ProTCLTrainer(
        model=model,
        device=device,
        config=config,
        logger=logger,
        timestamp=timestamp,
        run_name=args.name,
        use_wandb=args.use_wandb and is_master,
        use_amlt=args.amlt,
        loss_fn=loss_fn,
        is_master=is_master
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

    # Log sizes of all datasets
    [logger.info(f"{subset_name} dataset size: {len(dataset)}") for subset_name, subset in datasets.items() for dataset in subset]


    ####### TRAINING AND VALIDATION LOOPS #######
    if args.train_path_name is not None:
        # Train function
        Trainer.train(train_loader=loaders["train"][0],
            val_loader=loaders["validation"][0],
            train_eval_metrics=eval_metrics.get_metric_collection_with_regex(pattern="f1_m.*",
                                                                             threshold=0.5,
                                                                        num_labels=label_sample_sizes["train"] if (params['IN_BATCH_SAMPLING'] or params['GRID_SAMPLER']) is False else None
                                                                        ),
            val_eval_metrics=eval_metrics.get_metric_collection_with_regex(pattern="f1_m.*", threshold=0.5,
                                                                num_labels=label_sample_sizes["validation"]
                                                                ),
            val_optimization_metric_name=params["OPTIMIZATION_METRIC_NAME"])
    else:
        logger.info("Skipping training...")

    ####### TESTING LOOP #######
    all_test_metrics = {}
    all_metrics = {}


    # Setup for validation
    run_metrics = {'name':args.name}
    if args.save_val_test_metrics & is_master :
        metrics_results = read_json(args.save_val_test_metrics_path)
        
    if args.validation_path_name:
        # Reinitialize the validation loader with all the data, in case we were using a subset to expedite training
        logger.info(
            f"\n{'='*100}\nTesting on validation set\n{'='*100}")

        # Print the batch size used
        logger.info(f"Batch size: {params['TEST_BATCH_SIZE']}")
        if is_master:
            log_gpu_memory_usage(logger, 0)

        # Final validation using all labels
        torch.cuda.empty_cache()

        validation_metrics = Trainer.evaluate(
            data_loader=loaders["validation"][0],#full_val_loader,
            eval_metrics=eval_metrics.get_metric_collection_with_regex(pattern="f1_m.*",
                                                                    threshold=0.5,
                                                                    num_labels=label_sample_sizes["validation"]
                                                            ),
            save_results=args.save_prediction_results,
            metrics_prefix='final_validation'
                    )
        all_metrics.update(validation_metrics)
        logger.info(json.dumps(validation_metrics, indent=4))
        if args.save_val_test_metrics:
            run_metrics.update(validation_metrics)
        logger.info("Final validation complete.")

        
    # Setup for testing
    if args.test_paths_names:
        for idx, test_loader in enumerate(loaders["test"]):
            logger.info(
                f"\n{'='*100}\nTesting on test set {idx+1}/{len(loaders['test'])}\n{'='*100}")
            if is_master:
                log_gpu_memory_usage(logger, 0)

            # TODO: If best_val_th is not defined, alert an error to either provide a decision threshold or a validation datapath
            test_metrics = Trainer.evaluate(
                data_loader=test_loader,
                eval_metrics=eval_metrics.get_metric_collection_with_regex(pattern="f1_m.*",
                                                                          threshold=0.5,
                                                                          num_labels=label_sample_sizes["test"]),
                save_results=args.save_prediction_results,
                metrics_prefix=f'test_{idx+1}'
            )
            all_test_metrics.update(test_metrics)
            logger.info(json.dumps(test_metrics, indent=4))
            if args.save_val_test_metrics:
                run_metrics.update(test_metrics)
            logger.info("Testing complete.")

        all_metrics.update(test_metrics)



    ####### CLEANUP #######
    
    logger.info(
        f"\n{'='*100}\nTraining, validating, and testing COMPLETE\n{'='*100}\n")
    # W&B, MLFlow amd optional metric results saving
    if is_master:
        #Optionally save val/test results in json
        if args.save_val_test_metrics:
            metrics_results.append(run_metrics)
            write_json(metrics_results,args.save_val_test_metrics_path)
        # Log test metrics
        if args.test_paths_names:
            if args.use_wandb:
                wandb.log(all_test_metrics)
            if args.amlt & args.mlflow:
                mlflow.log_metrics(all_test_metrics)
        
        # Log val metrics
        if args.validation_path_name:
            if args.use_wandb:
                wandb.log(validation_metrics)
            if args.amlt & args.mlflow:
                mlflow.log_metrics(validation_metrics)
    
        '''
        #Create wandb summary table
        wandb.log({{'':args.name},
                   all_metrics
                   })
        '''
        #Close metric loggers
        if args.use_wandb:
            wandb.finish()
        if args.amlt & args.mlflow:
            mlflow.end_run()
        
    # Loggers
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    # Torch
    torch.cuda.empty_cache()
    dist.destroy_process_group()


"""
 Sample usage (single GPU): 
 python main.py 
    --nodes 1 
    --gpus 1 
    --nr 0 
    --train-path-name TRAIN_DATA_PATH 
    --validation-path-name VAL_DATA_PATH 
    --full-path-name FULL_DATA_PATH 
    --test-paths-names TEST_DATA_PATH
    --use-wandb
"""
if __name__ == "__main__":
    main()
