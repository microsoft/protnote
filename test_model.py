import argparse
import subprocess
from itertools import product




MODEL_PATH_TOKEN = '<MODEL_PATH>'
MODEL_NAME_TOKEN = '<MODEL_NAME>'
TEST_COMMANDS = {
    "TEST_DATA_PATH_ZERO_SHOT": f"python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT_{MODEL_NAME_TOKEN}",
    "TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES": f"python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES_{MODEL_NAME_TOKEN}",
    "TEST_EC_DATA_PATH_ZERO_SHOT": f"python main.py --test-paths-names TEST_EC_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct DECISION_TH .3 --load-model {MODEL_PATH_TOKEN} --annotations-path-name EC_ANNOTATIONS_PATH --base-label-embedding-name EC_BASE_LABEL_EMBEDDING_PATH --name TEST_EC_DATA_PATH_ZERO_SHOT_{MODEL_NAME_TOKEN}",
    "TEST_2024_DATA_PATH": f"python main.py --test-paths-names TEST_2024_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_2024_DATA_PATH_{MODEL_NAME_TOKEN}",
    "TEST_2024_PINF_VOCAB_DATA_PATH": f"python main.py --test-paths-names TEST_2024_PINF_VOCAB_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_2024_PINF_VOCAB_DATA_PATH_{MODEL_NAME_TOKEN}",
    "TEST_DATA_PATH": f"python main.py --test-paths-names TEST_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --name TEST_DATA_PATH_{MODEL_NAME_TOKEN}",
    "VAL_DATA_PATH": f"python main.py --test-paths-names VAL_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --name VAL_DATA_PATH_{MODEL_NAME_TOKEN}"
}


def main():
    parser = argparse.ArgumentParser(description="Run specified test paths.")
    parser.add_argument("--test-paths-names", nargs='+', default = [],required=False, help="List of test path names to run")
    parser.add_argument("--model-path",type=str, required=True, help='The path to the model we are testing')
    parser.add_argument("--test-type",type=str, default='all', help='One of all, baseline, model')
    parser.add_argument("--save-prediction-results",action='store_true',default=False,required=False)
    parser.add_argument("--save-embeddings",action='store_true',default=False,required=False)
    parser.add_argument("--save-val-test-metrics",action='store_true',default=False,required=False)
    parser.add_argument("--save-val-test-metrics-path", default = 'outputs/results/ablation_results.json',
                        help=" Path to append val/test metrics to json",type=str) 
    
    args = parser.parse_args()

     
    
    model_name = args.model_path.split('.pt')[0].split('/')[-1]
    UPDATED_TEST_COMMANDS = {k:v.replace(MODEL_PATH_TOKEN,args.model_path)
                                .replace(MODEL_NAME_TOKEN,model_name)
                                + (' --save-prediction-results' if args.save_prediction_results else '')\
                                + (' --save-embeddings' if args.save_embeddings else '')\
                                + (' --save-val-test-metrics' if args.save_val_test_metrics else '')\
                                + ' --save-val-test-metrics-path ' +  args.save_val_test_metrics_path
                             for k,v in TEST_COMMANDS.items()}
    

    args.test_paths_names = list(UPDATED_TEST_COMMANDS.keys()) if args.test_paths_names==[] else args.test_paths_names
    

    if args.test_type in ['all','model']:
        print('testing models...')
        for test_path in args.test_paths_names:
            if test_path in UPDATED_TEST_COMMANDS:
                command = UPDATED_TEST_COMMANDS[test_path]
                print(f"Running command for {test_path}: {command}")
                subprocess.run(command, shell=True)
            else:
                print(f"Test path name {test_path} not found in TEST_COMMANDS dictionary.")

    if args.test_type in ['all','baseline']:
        print('Running baselines')    
        for label_encoder,test_set in list(product(['BioGPT','E5'],
                                                ['TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES','TEST_DATA_PATH_ZERO_SHOT'])
                                            ):
            command = f"python run_baseline.py --test-name {test_set} --cache --model-name {model_name} --label-embedding-model {label_encoder} --annotation-type GO"
            print(f"Running command: {command}")
            subprocess.run(command, shell=True)

        for label_encoder,test_set in list(product(['BioGPT','E5'],
                                                ['TEST_EC_DATA_PATH_ZERO_SHOT'])
                                            ):
            command = f"python run_baseline.py --test-name {test_set} --cache --model-name {model_name} --label-embedding-model {label_encoder} --annotation-type EC"
            print(f"Running command: {command}")
            subprocess.run(command, shell=True)

if __name__ == "__main__":
    

    
    main()
