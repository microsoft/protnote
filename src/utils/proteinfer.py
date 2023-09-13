from src.utils.data import read_pickle
import numpy as np
import collections
import torch


def transfer_tf_weights_to_torch(torch_model: torch.nn.Module, tf_weights_path: str):
    # Load tensorflow variables. Remove global step variable and add it as num_batches variable for each batchnorm
    tf_weights = read_pickle(tf_weights_path)
    # total training steps from the paper. Used for batch norm running statistics.
    num_batches = tf_weights['inferrer/global_step:0']
    tf_weights.pop('inferrer/global_step:0')
    temp = {}
    for (tf_name, tf_param) in tf_weights.items():
        temp[tf_name] = tf_param
        if ('batch_normalization' in tf_name) & ('moving_variance' in tf_name):
            num_batches_name = '/'.join(tf_name.split('/')
                                        [:-1]+['num_batches_tracked:0'])
            temp[num_batches_name] = np.array(num_batches)
    tf_weights = temp

    # Get pytorch model variables
    state_dict = torch_model.state_dict()
    state_dict_list = [(k, v) for k, v in state_dict.items()]

    print('Pytorch num variables:', len(state_dict))
    print('Tensorflow num variables:', len(tf_weights))
    print('\n')
    with torch.no_grad():
        for (name, param), (tf_name, tf_param) in zip(state_dict_list, tf_weights.items()):

            if (tf_param.ndim >= 2):
                tf_param = np.transpose(tf_param,
                                        tuple(
                                            sorted(range(tf_param.ndim), reverse=True))
                                        )

            print(f'{name}:{param.shape}', '<-->',
                  f'{tf_name}:{tf_param.shape}')
            assert tf_param.shape == param.detach().numpy(
            ).shape, f"{name} and {tf_name} don't have the same shape"
            state_dict[name] = torch.from_numpy(tf_param)

    torch_model.load_state_dict(state_dict)


def reverse_map(
        applicable_label_dict,
        label_vocab=None):
    """Flip parenthood dict to map parents to children.

    Args:
      applicable_label_dict: e.g. output of get_applicable_label_dict.
      label_vocab: e.g. output of inference_lib.vocab_from_model_base_path

    Returns:
      collections.defaultdict of k, v where:
      k: originally the values in applicable_label_dict
      v: originally the keys in applicable_label_dict.
      The defaultdict returns an empty frozenset for keys that are not found.
      This behavior is desirable for lifted clan label normalizers, where
      keys may not imply themselves.
    """
    # This is technically the entire transitive closure, so it is safe for DAGs
    # (e.g. GO labels).

    children = collections.defaultdict(set)
    for child, parents in applicable_label_dict.items():
        # Avoid adding children which don't appear in the vocab.
        if label_vocab is None or child in label_vocab:
            for parent in parents:
                children[parent].add(child)
    children = {k: frozenset(v) for k, v in children.items()}
    return collections.defaultdict(frozenset, children.items())


def normalize_confidences(
        predictions,
        label_vocab,
        applicable_label_dict):
    """Set confidences of parent labels to the max of their children.

    Args:
      predictions: [num_sequences, num_labels] ndarray.
      label_vocab: list of vocab strings in an order that corresponds to
        `predictions`.
      applicable_label_dict: Mapping from labels to their parents (including
        indirect parents).

    Returns:
      A numpy array [num_sequences, num_labels] with confidences where:
      if label_vocab[k] in applicable_label_dict[label_vocab[j]],
      then arr[i, j] >= arr[i, k] for all i.
    """
    vocab_indices = {v: i for i, v in enumerate(label_vocab)}
    children = reverse_map(applicable_label_dict,
                           set(vocab_indices.keys()))

    # Only vectorize this along the sequences dimension as the number of children
    # varies between labels.
    label_confidences = []
    for label in label_vocab:
        child_indices = np.array(
            [vocab_indices[child] for child in children[label]])
        if child_indices.size > 1:
            confidences = np.max(predictions[:, child_indices], axis=1)
            label_confidences.append(confidences)
        else:
            label_confidences.append(predictions[:, vocab_indices[label]])

    return np.stack(label_confidences, axis=1)
