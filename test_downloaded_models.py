from src.utils.data import read_yaml
import glob
import re
import os

amlt_config = read_yaml('amlt_config.yaml')

jobs = amlt_config['jobs']
for idx,job in enumerate(jobs):
    command = job['command'][0]\
        .replace('--nodes 1 --gpus 8 --train-path-name TRAIN_DATA_PATH_ZERO_SHOT ','')\
        .replace(' --amlt --use-wandb','')\
        .replace('ESTIMATE_MAP True','ESTIMATE_MAP False')


    model_name = re.search(r"--name\s+'([^']*)'", command)
    # Extract the model_nameed group if present
    if model_name:
        extracted_string = model_name.group(1)
        print('='*20)
        print('Found model name:')
        print(extracted_string)
        print('='*20)

    else:
        print('='*20)
        print('NO MODEL FOUND...SKIPPING')
        print('='*20)
        continue
    
    compatible_model_paths = glob.glob(f'data/models/ProTCL/*{extracted_string}*.pt')

    assert len(glob.glob(f'data/models/ProTCL/*{extracted_string}*.pt'))==1, "More than 1 model found"

    model_path = glob.glob(f'data/models/ProTCL/*{extracted_string}*.pt')[0][5:]

    print('='*20)
    print('Found model path:')
    print(model_path)
    print('='*20)

    command+=f" --load-model '{model_path}' --save-val-test-metrics"

    print(f"executing the following command: {command}\n\n")
    os.system(command)
    print(f'Finished job #{idx}')