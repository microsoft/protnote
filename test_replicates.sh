
numbers=(12 22 32 42 52) 

for number in "${numbers[@]}"; do
    file_name="models/ProtNote/seed_replicates_v9_${number}_sum_last_epoch.pt"
    python test_model.py --test-paths-names "TEST_DATA_PATH_ZERO_SHOT" "TEST_DATA_PATH_ZERO_SHOT_LEAF_NODES" "TEST_EC_DATA_PATH_ZERO_SHOT" "TEST_DATA_PATH" --model-path "${file_name}" --save-prediction-results --test-type "model"
done


