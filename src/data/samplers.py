import random
from itertools import product
import numpy as np
from torch.utils.data import BatchSampler

from torch.utils.data import Sampler
from typing import Optional
import math
import torch
from torch.utils.data.distributed import DistributedSampler

class GeneralDistributedSampler(DistributedSampler):

    """
    Class to use distributed sampler with any sampler!
    """

    def __init__(self, sampler: Sampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 seed: int = 0, drop_last: bool = False):
        
        #Same as normal DistributedSampler with shuffle = False
        super().__init__(dataset = sampler,
                         num_replicas=num_replicas,
                         rank=rank,
                         shuffle=False,
                         seed = seed,
                         drop_last=drop_last)
        
        assert len(sampler)>num_replicas, "Total samples must be > num replicas"
        
    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch+self.seed)
        indices = list(self.dataset)
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
        

class GridBatchSampler(BatchSampler):

    def __init__(self,
                 observation_sampler,
                 observations_batch_size,
                 drop_last_observation_batch,
                 num_labels,
                 labels_batch_size,
                 shuffle_grid = True):
        
        self.observation_sampler = observation_sampler
        self.observations_batch_size = observations_batch_size
        self.drop_last_observation_batch = drop_last_observation_batch

        self.num_labels = num_labels
        self.labels_batch_size = labels_batch_size
        self.shuffle_grid = shuffle_grid
        self.labels_idxs = list(range(num_labels))
        self.calculate_num_batches()
        
    def __iter__(self):
        random.shuffle(self.labels_idxs)
        print('Getting label batches...')
        observation_batches = self.get_observation_batches()
        print('Done...')

        print('Getting observation batches...')
        label_batches = self.get_label_batches()
        print('Done...')

        print('Getting combinations...')
        obs_labels_batch_combinations = list(product(observation_batches,label_batches))

        print('Done...')
        
        if self.shuffle_grid:
            print('Shuffling...')
            random.shuffle(obs_labels_batch_combinations)
        print('Done...')
        for observation_batch,label_batch in obs_labels_batch_combinations:
            yield list(product(observation_batch, [label_batch]))#[observation_batch,label_batch]
    
    def calculate_num_batches(self):
        
        num_label_batches = np.ceil(self.num_labels/self.labels_batch_size)
        num_observation_batches = (np.ceil(len(self.observation_sampler)/self.observations_batch_size)
                                   if not self.drop_last_observation_batch
                                   else len(self.observation_sampler)//self.observations_batch_size)
        print('Done...')

        self.total_num_batches = int(num_label_batches*num_observation_batches)

    def __len__(self):
        return self.total_num_batches
    

    def get_label_batches(self):

        #n_chunks = int(np.ceil(self.num_labels/self.labels_batch_size))
        return [self.labels_idxs[i:i+self.labels_batch_size] for i in range(0,self.num_labels,self.labels_batch_size)]
        

    def get_observation_batches(self):

        batches = []

        if self.drop_last_observation_batch:
            observation_sampler_iter = iter(self.observation_sampler)
            while True:
                try:
                    batch = [next(observation_sampler_iter) for _ in range(self.observations_batch_size)]
                    batches.append(batch)
                except StopIteration:
                    break
        else:
            batch = [0] * self.observations_batch_size
            idx_in_batch = 0
            for idx in self.observation_sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.observations_batch_size:
                    batches.append(batch)
                    idx_in_batch = 0
                    batch = [0] * self.observations_batch_size
            if idx_in_batch > 0:
                batches.append(batch[:idx_in_batch])
        return batches

