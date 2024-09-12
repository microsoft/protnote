import random
from itertools import product
import numpy as np
from torch.utils.data import BatchSampler
from torch.utils.data import Sampler, WeightedRandomSampler
from typing import Optional
import math
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math
from torch.utils.data import Dataset


class GeneralDistributedSampler(DistributedSampler):

    """
    Class to use distributed sampler with any sampler!
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ):
        # Same as normal DistributedSampler with shuffle = False
        super().__init__(
            dataset=sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=drop_last,
        )

        assert len(sampler) > num_replicas, "Total samples must be > num replicas"

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch + self.seed)
        indices = list(self.dataset)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, world_size=None, rank=None, replacement=True):
        # Get the world size and rank if not provided
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.weights = weights
        self.world_size = world_size
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement

        # Ensure weights is a tensor
        if not isinstance(self.weights, torch.Tensor):
            self.weights = torch.tensor(self.weights, dtype=torch.double)

        # Determine the number of samples for each GPU, rounding down to ensure it is evenly divisible
        self.num_samples = int(math.floor(len(self.weights) * 1.0 / self.world_size))

        # Determine the total number of samples
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        # Shuffle based on the epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # Create a weighted sample for the entire dataset
        if self.replacement:
            indices = torch.multinomial(
                self.weights, self.total_size, replacement=True, generator=g
            )
        else:
            assert (
                len(self.weights) > self.total_size
            ), "When sampling without replacement, number of samples to draw must be less than the number of elements in the dataset"
            indices = torch.multinomial(
                self.weights, self.total_size, replacement=False, generator=g
            )

        # Subsample for the current process
        indices_for_one_gpu = indices[self.rank : self.total_size : self.world_size]

        # Shuffle each epoch
        indices_for_one_gpu = indices_for_one_gpu[
            torch.randperm(len(indices_for_one_gpu), generator=g)
        ].tolist()

        return iter(indices_for_one_gpu)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class GridBatchSampler(BatchSampler):
    def __init__(
        self,
        observation_sampler,
        observations_batch_size,
        drop_last_observation_batch,
        num_labels,
        labels_batch_size,
        shuffle_grid=True,
    ):
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
        print("Getting label batches...")
        observation_batches = self.get_observation_batches()
        print("Done...")

        print("Getting observation batches...")
        label_batches = self.get_label_batches()
        print("Done...")

        print("Getting combinations...")
        obs_labels_batch_combinations = list(
            product(observation_batches, label_batches)
        )

        print("Done...")

        if self.shuffle_grid:
            print("Shuffling...")
            random.shuffle(obs_labels_batch_combinations)
        print("Done...")
        for observation_batch, label_batch in obs_labels_batch_combinations:
            yield list(
                product(observation_batch, [label_batch])
            )  # [observation_batch,label_batch]

    def calculate_num_batches(self):
        num_label_batches = np.ceil(self.num_labels / self.labels_batch_size)
        num_observation_batches = (
            np.ceil(len(self.observation_sampler) / self.observations_batch_size)
            if not self.drop_last_observation_batch
            else len(self.observation_sampler) // self.observations_batch_size
        )
        print("Done...")

        self.total_num_batches = int(num_label_batches * num_observation_batches)
        print(
            f"num label batches = {num_label_batches}, num observation batches = {num_observation_batches}"
        )
        print(f"total batches = {self.total_num_batches}")

    def __len__(self):
        return self.total_num_batches

    def get_label_batches(self):
        # n_chunks = int(np.ceil(self.num_labels/self.labels_batch_size))
        return [
            self.labels_idxs[i : i + self.labels_batch_size]
            for i in range(0, self.num_labels, self.labels_batch_size)
        ]

    def get_observation_batches(self):
        batches = []

        if self.drop_last_observation_batch:
            observation_sampler_iter = iter(self.observation_sampler)
            while True:
                try:
                    batch = [
                        next(observation_sampler_iter)
                        for _ in range(self.observations_batch_size)
                    ]
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


def observation_sampler_factory(
    distribute_labels: bool,
    weighted_sampling: bool,
    shuffle: bool,
    dataset: Dataset = None,
    world_size: int = 1,
    rank: int = 0,
    sequence_weights: torch.Tensor = None,
):
    if distribute_labels and not weighted_sampling:
        print("WARNING: No Sampler used for distribute labels")
        sampler = None
    elif not distribute_labels and world_size == 1 and weighted_sampling:
        # If NOT distributing labels, and not training on multiple GPU's, create a non-distributed weighted sampler with replacement
        assert sequence_weights is not None, "Weighted RandomSampler requires weights"

        sampler = WeightedRandomSampler(
            sequence_weights, len(sequence_weights), replacement=True
        )
    elif not distribute_labels and world_size > 1 and weighted_sampling:
        # If distributing sequences across multiple GPUs with a weighted sampler, create custom DistributedWeightedSampler
        sampler = DistributedWeightedSampler(
            sequence_weights,
            world_size=world_size,
            rank=rank,
            replacement=True,
        )
    elif not distribute_labels and not weighted_sampling:
        # If simply distributing sequences across GPU's without weighted sampling, use a distributed sampler

        assert dataset is not None, "DistributeSampler requires dataset"

        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
    else:
        # Raise error
        raise ValueError(
            "Invalid combination of WEIGHTED_SAMPLING, WORLD_SIZE, and DISTRIBUTE_LABELS parameters"
        )

    return sampler
