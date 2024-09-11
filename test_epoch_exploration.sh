epochs=(10 20 30 40) 

for epoch in "${epochs[@]}"; do
    file_name="models/ProtNote/2024-08-12_09-04-06_seed_replicates_v9_42_mean_epoch_${epoch}.pt"
    # python test_model.py --test-paths-names "TEST_DATA_PATH_ZERO_SHOT" "TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES" "TEST_EC_DATA_PATH_ZERO_SHOT" "TEST_DATA_PATH" "VAL_DATA_PATH" --model-path "${file_name}" --save-prediction-results --save-val-test-metrics --test-type "model" --save-val-test-metrics-path 'outputs/results/epoch_exploration.json'
done
