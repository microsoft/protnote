import argparse
import subprocess

MODEL_PATH_TOKEN = '<MODEL_PATH>'
TEST_COMMANDS = {
    "TEST_DATA_PATH_ZERO_SHOT": f"python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT --save-prediction-results",
    "TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES": f"python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES --save-prediction-results",
    "TEST_EC_DATA_PATH_ZERO_SHOT": f"python main.py --test-paths-names TEST_EC_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct DECISION_TH .3 --load-model {MODEL_PATH_TOKEN} --annotations-path-name EC_ANNOTATIONS_PATH --base-label-embedding-name EC_BASE_LABEL_EMBEDDING_PATH --name TEST_EC_DATA_PATH_ZERO_SHOT --save-prediction-results",
    "TEST_2024_DATA_PATH": f"python main.py --test-paths-names TEST_2024_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_2024_DATA_PATH --save-prediction-results",
    "TEST_2024_PINF_VOCAB_DATA_PATH": f"python main.py --test-paths-names TEST_2024_PINF_VOCAB_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_2024_PINF_VOCAB_DATA_PATH --save-prediction-results",
    "TEST_DATA_PATH": f"python main.py --test-paths-names TEST_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --name TEST_DATA_PATH --save-prediction-results"
}

def main():
    MODEL_PATH_TOKEN = '<MODEL_PATH>'
    TEST_COMMANDS = {
        "TEST_DATA_PATH_ZERO_SHOT": f"python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT --save-prediction-results",
        "TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES": f"python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES --save-prediction-results",
        "TEST_EC_DATA_PATH_ZERO_SHOT": f"python main.py --test-paths-names TEST_EC_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct DECISION_TH .3 --load-model {MODEL_PATH_TOKEN} --annotations-path-name EC_ANNOTATIONS_PATH --base-label-embedding-name EC_BASE_LABEL_EMBEDDING_PATH --name TEST_EC_DATA_PATH_ZERO_SHOT --save-prediction-results",
        "TEST_2024_DATA_PATH": f"python main.py --test-paths-names TEST_2024_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_2024_DATA_PATH --save-prediction-results",
        "TEST_2024_PINF_VOCAB_DATA_PATH": f"python main.py --test-paths-names TEST_2024_PINF_VOCAB_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_2024_PINF_VOCAB_DATA_PATH --save-prediction-results",
        "TEST_DATA_PATH": f"python main.py --test-paths-names TEST_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model {MODEL_PATH_TOKEN} --name TEST_DATA_PATH --save-prediction-results"
    }
    parser = argparse.ArgumentParser(description="Run specified test paths.")
    parser.add_argument("--test-paths-names", nargs='+', default = [],required=False, help="List of test path names to run")
    parser.add_argument("--model-path",type=str, required=True, help='The path to the model we are testing')
    
    args = parser.parse_args()
    
    TEST_COMMANDS = {k:v.replace(MODEL_PATH_TOKEN,args.model_path) for k,v in TEST_COMMANDS.items()}

    args.test_paths_names = list(TEST_COMMANDS.keys()) if args.test_paths_names==[] else args.test_paths_names
    
    for test_path in args.test_paths_names:
        if test_path in TEST_COMMANDS:
            command = TEST_COMMANDS[test_path]
            print(f"Running command for {test_path}: {command}")
            subprocess.run(command, shell=True)
        else:
            print(f"Test path name {test_path} not found in TEST_COMMANDS dictionary.")

if __name__ == "__main__":
    

    
    main()
