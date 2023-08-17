from src.utils.data import read_pickle
import numpy as np
import torch

def transfer_tf_weights_to_torch(torch_model:torch.nn.Module, tf_weights_path:str):
    # Load tensorflow variables. Remove global step variable and add it as num_batches variable for each batchnorm
    tf_weights = read_pickle(tf_weights_path)
    num_batches = tf_weights['inferrer/global_step:0'] #total training steps from the paper. Used for batch norm running statistics.
    tf_weights.pop('inferrer/global_step:0')
    temp = {}
    for (tf_name, tf_param) in tf_weights.items():
        temp[tf_name] = tf_param
        if ('batch_normalization' in tf_name) & ('moving_variance' in tf_name):
            num_batches_name = '/'.join(tf_name.split('/')[:-1]+['num_batches_tracked:0'])
            temp[num_batches_name] = np.array(num_batches)
    tf_weights = temp


    # Get pytorch model variables
    state_dict = torch_model.state_dict()
    state_dict_list = [(k,v) for k,v in state_dict.items()]

    print('Pytorch num variables:',len(state_dict))
    print('Tensorflow num variables:',len(tf_weights))
    print('\n')
    with torch.no_grad():
        for (name, param), (tf_name, tf_param) in zip(state_dict_list, tf_weights.items()):
            
            if (tf_param.ndim>=2):
                tf_param =np.transpose(tf_param,
                                    tuple(sorted(range(tf_param.ndim),reverse=True))
                                    ) 
            
            print(f'{name}:{param.shape}','<-->',f'{tf_name}:{tf_param.shape}')
            assert tf_param.shape == param.detach().numpy().shape, f"{name} and {tf_name} don't have the same shape"
            state_dict[name] = torch.from_numpy(tf_param)
    
    torch_model.load_state_dict(state_dict)