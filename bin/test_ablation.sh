names=(
    "2024-08-21_06-00-55_seed_42_ablations_OVERRIDE_WEIGHTED_SAMPLING_False_last_epoch.pt" \
    "2024-08-21_06-05-49_seed_42_ablations_OVERRIDE_LABEL_EMBEDDING_NOISING_ALPHA_0_last_epoch.pt" \
    "2024-08-21_06-12-36_seed_42_ablations_OVERRIDE_LOSS_FN_BCE_last_epoch.pt" \
    "2024-08-21_06-33-42_seed_42_ablations_OVERRIDE_LABEL_AUGMENTATION_DESCRIPTIONS_name_last_epoch.pt" \
    "2024-08-21_06-43-50_seed_42_ablations_OVERRIDE_LABEL_ENCODER_CHECKPOINT_biogpt_last_epoch.pt" \
    "2024-08-21_06-00-52_seed_42_ablations_OVERRIDE_AUGMENT_RESIDUE_PROBABILITY_0_last_epoch.pt" \
    "2024-09-09_19-34-44_seed_12_ablations_OVERRIDE_LABEL_AUGMENTATION_DESCRIPTIONS_name_last_epoch.pt" \
    "2024-09-09_19-48-03_seed_12_ablations_OVERRIDE_LABEL_ENCODER_CHECKPOINT_biogpt_last_epoch.pt" \
    "2024-09-09_19-47-36_seed_12_ablations_OVERRIDE_WEIGHTED_SAMPLING_False_last_epoch.pt" \
    "2024-09-09_19-48-10_seed_12_ablations_OVERRIDE_AUGMENT_RESIDUE_PROBABILITY_0_last_epoch.pt" \
    "2024-09-09_19-48-17_seed_12_ablations_OVERRIDE_LABEL_EMBEDDING_NOISING_ALPHA_0_last_epoch.pt" \
    "2024-09-09_19-43-24_seed_12_ablations_OVERRIDE_LOSS_FN_BCE_last_epoch.pt" \
    "2024-09-09_19-48-10_seed_22_ablations_OVERRIDE_LOSS_FN_BCE_last_epoch.pt" \
    "2024-09-09_19-44-01_seed_22_ablations_OVERRIDE_WEIGHTED_SAMPLING_False_last_epoch.pt" \
    "2024-09-09_19-43-36_seed_22_ablations_OVERRIDE_LABEL_ENCODER_CHECKPOINT_biogpt_last_epoch.pt" \
    "2024-09-09_19-48-59_seed_22_ablations_OVERRIDE_AUGMENT_RESIDUE_PROBABILITY_0_last_epoch.pt" \
    "2024-09-09_19-44-49_seed_22_ablations_OVERRIDE_LABEL_AUGMENTATION_DESCRIPTIONS_name_last_epoch.pt" \
    "2024-09-09_19-47-49_seed_22_ablations_OVERRIDE_LABEL_EMBEDDING_NOISING_ALPHA_0_last_epoch.pt"
    ) 

output_file='final_model_ablations.json'
for name in "${names[@]}"; do
    model_file="${name}"
    model_name="${name%.pt}"
    
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
    
    python bin/main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT "$label_encoder_checkpoint" AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS "$description_augmentation" INFERENCE_GO_DESCRIPTIONS "$description_augmentation" --model-file "$model_file" --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT_"$model_name" --save-val-test-metrics --save-val-test-metrics-file "$output_file"
    python bin/main.py --test-paths-names TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.3 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT "$label_encoder_checkpoint" AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS "$description_augmentation" INFERENCE_GO_DESCRIPTIONS "$description_augmentation" --model-file "$model_file" --base-label-embedding-name GO_2024_BASE_LABEL_EMBEDDING_PATH --name TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES_"$model_name" --save-val-test-metrics --save-val-test-metrics-file "$output_file"
    python bin/main.py --test-paths-names VAL_DATA_PATH --override EXTRACT_VOCABULARIES_FROM null DECISION_TH 0.5 ESTIMATE_MAP False OPTIMIZATION_METRIC_NAME f1_macro LABEL_ENCODER_CHECKPOINT "$label_encoder_checkpoint" AUGMENT_RESIDUE_PROBABILITY 0.1 LABEL_EMBEDDING_NOISING_ALPHA 20 TEST_BATCH_SIZE 8 LABEL_AUGMENTATION_DESCRIPTIONS "$description_augmentation" INFERENCE_GO_DESCRIPTIONS "$description_augmentation" --model-file "$model_file" --name VAL_DATA_PATH_"$model_name" --save-val-test-metrics --save-val-test-metrics-file "$output_file"
done