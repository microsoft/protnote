ids=(13703706 13703742 13703997 13704131 13705631)  # 13705668 13705677 13705689 13705708 13705728 )
eval "$(conda shell.bash hook)"
conda activate proteinfer
for id in "${ids[@]}"; do
    if [ ! -e "data/models/proteinfer/GO_model_weights${id}.pkl" ]; then
        wget https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-${id}.tar.gz
        tar -xvzf noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-${id}.tar.gz
        mv -f noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-${id} proteinfer/cached_models/
        python proteinfer/export_proteinfer.py --model-path proteinfer/cached_models/noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-${id} --model-name GO --add-model-id
        mv proteinfer/export/GO_model_weights${id}.pkl data/models/proteinfer/GO_model_weights${id}.pkl
    else
        echo "${id} weights already exist"
    fi
done    
conda deactivate
conda activate protein_functions_310

for id in "${ids[@]}"; do
    python test_proteinfer.py --test-paths-names TEST_DATA_PATH --only-inference --only-represented-labels --save-prediction-results --name TEST_DATA_PATH_proteinfer --model-weights-id ${id}
done

conda deactivate