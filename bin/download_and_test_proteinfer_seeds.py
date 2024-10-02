import os
import subprocess
import argparse
from pathlib import Path

# Define the list of IDs
def run_command(command):
    """Utility function to run a shell command"""
    result = subprocess.run(command, shell=True, check=True, text=True)
    return result

def download_and_extract_model(id):
    """Download, extract, and export the model if it doesn't exist"""
    model_dir = Path(__file__).resolve().parent.parent / "data" / "models" / "proteinfer" 
    model_file = model_dir / f"GO_model_weights{id}.pkl"
    if not os.path.exists(model_file):
        tar_file = f"noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-{id}.tar.gz"
        download_url = f"https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/{tar_file}"
        
        # Download the model
        run_command(f"wget {download_url}")
        
        # Extract the tar file
        run_command(f"tar -xvzf {tar_file}")
        
        # Move extracted model to cached models directory
        model_id_dir = f"noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-{id}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        run_command(f"mv -f {model_id_dir} {model_dir}")
        
        # Run the export script
        run_command(f"conda run -n proteinfer python bin/export_proteinfer.py --model-path {model_dir / model_id_dir} --output-dir {model_dir} --model-name GO --add-model-id")

        #Clean up. Orginal weights are not needed.
        
        run_command(f"rm -rf {model_dir / model_id_dir}")
    
    else:
        print(f"{id} weights already exist")

def run_inference(id):
    """Run inference with the specified model weights"""
    run_command(f"conda run -n protnote python bin/test_proteinfer.py --test-paths-names TEST_DATA_PATH --only-inference --only-represented-labels --save-prediction-results --name TEST_DATA_PATH_proteinfer --model-weights-id {id}")

def main(ids:list,get_predictions:bool):

    # Loop over the IDs and handle the models
    for id in ids:
        download_and_extract_model(id)
    
    # Run inference on the models
    if get_predictions:
        for id in ids:
            run_inference(id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run protein inference on multiple models")
    parser.add_argument(
        "--ids",
        nargs="+",
        default=[13703706, 13703742, 13703997, 13704131, 13705631],
        required=False,
        help="List of proteinfer model ids",
    )
    parser.add_argument(
        "--get-predictions",
        action="store_true",
        default=False,
        help="Whether to run inference and store the predictions of the ProteInfer models",
    )

    args = parser.parse_args()
    main(ids = args.ids, get_predictions = args.get_predictions)
