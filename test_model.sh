
# Ours GO Zero Shot 2024 new seqs + new labels + added vocab terms
python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model 'models/ProTCL/normal_test_label_aug_v4.pt' --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH

# Ours GO Zero Shot 2024 Leaf Nodes. new seqs + new labels only leaf nodes + added vocab terms
python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model 'models/ProTCL/normal_test_label_aug_v4.pt' --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH

# Ours EC Numbers.
python main.py --test-paths-names TEST_EC_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct DECISION_TH .3 --load-model "models/ProTCL/normal_test_label_aug_v4.pt" --annotations-path-name EC_ANNOTATIONS_PATH --base-label-embedding-name EC_BASE_LABEL_EMBEDDING_PATH

# Ours Updated Supervised Test Set. pinf test seqs + new labels + new vocab
python main.py --test-paths-names TEST_2024_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model 'models/ProTCL/normal_test_label_aug_v4.pt' --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH

# Ours Updated Supervised Test Set. pinf test seqs + new labels + pinf/old vocab
python main.py --test-paths-names TEST_2024_PINF_VOCAB_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model 'models/ProTCL/normal_test_label_aug_v4.pt' --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH

# Ours Proteinfer Supervised Test Set. pinf test seqs + old/pinf labels + old/pinf vocab
python main.py --test-paths-names TEST_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT intfloat/multilingual-e5-large-instruct AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 64 LABEL_AUGMENTATION_DESCRIPTIONS name+label INFERENCE_GO_DESCRIPTIONS name+label --load-model 'models/ProTCL/normal_test_label_aug_v4.pt'