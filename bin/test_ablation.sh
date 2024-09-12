names=(
    "2024-08-21_06-00-55_SEED_42_WEIGHTED_SAMPLING_False_SEQUENCE_WEIGHT_AGG_sum_last_epoch.pt" \
    "2024-08-21_06-05-49_SEED_42_LABEL_EMBEDDING_NOISING_ALPHA_0_SEQUENCE_WEIGHT_AGG_sum_last_epoch.pt" \
    "2024-08-21_06-12-36_SEED_42_LOSS_FN_BCE_SEQUENCE_WEIGHT_AGG_sum_last_epoch.pt" \
    "2024-08-21_06-33-42_SEED_42_LABEL_AUGMENTATION_DESCRIPTIONS_name_SEQUENCE_WEIGHT_AGG_sum_last_epoch.pt" \
    "2024-08-21_06-43-50_SEED_42_LABEL_ENCODER_CHECKPOINT_biogpt_SEQUENCE_WEIGHT_AGG_sum_last_epoch.pt" \
    "2024-08-21_06-00-52_SEED_42_AUGMENT_RESIDUE_PROBABILITY_0_SEQUENCE_WEIGHT_AGG_sum_last_epoch.pt"
    ) 

output_path='outputs/results/final_model_ablations_part2.json'
for name in "${names[@]}"; do
    file_name="models/ProtNote/${name}"
    model_name="${name%.pt}"

    # python test_model.py --test-paths-names "TEST_DATA_PATH_ZERO_SHOT" "TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES" "TEST_EC_DATA_PATH_ZERO_SHOT" "VAL_DATA_PATH" --model-path "${file_name}" --save-val-test-metrics --test-type "model" --save-val-test-metrics-path "$output_path"
    
    if [[ "$model_name" == *"LABEL_AUGMENTATION_DESCRIPTIONS_name"* ]]
    then  
        description_augmentation="name"
    else
        description_augmentation="name+label"
    fi 
    
    if [[ "$model_name" == *"biogpt"* ]]
    then  
        label_encoder_checkpoint="microsoft/biogpt"        
    else
        label_encoder_checkpoint="intfloat/multilingual-e5-large-instruct"
    fi 
    
    python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT "$label_encoder_checkpoint" AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS "$description_augmentation" INFERENCE_GO_DESCRIPTIONS "$description_augmentation" --load-model "$file_name" --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT_"$model_name" --save-val-test-metrics --save-val-test-metrics-path "$output_path"
    python main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT "$label_encoder_checkpoint" AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS "$description_augmentation" INFERENCE_GO_DESCRIPTIONS "$description_augmentation" --load-model "$file_name" --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES_"$model_name" --save-val-test-metrics --save-val-test-metrics-path "$output_path"
    python main.py --test-paths-names TEST_EC_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT "$label_encoder_checkpoint" DECISION_TH .3 --load-model "$file_name" --annotations-path-name EC_ANNOTATIONS_PATH --base-label-embedding-name EC_BASE_LABEL_EMBEDDING_PATH --name TEST_EC_DATA_PATH_ZERO_SHOT_"$model_name" --save-val-test-metrics --save-val-test-metrics-path "$output_path"
    python main.py --test-paths-names VAL_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT "$label_encoder_checkpoint" AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS "$description_augmentation" INFERENCE_GO_DESCRIPTIONS "$description_augmentation" --load-model "$file_name" --name VAL_DATA_PATH_"$model_name" --save-val-test-metrics --save-val-test-metrics-path "$output_path"
done