"""
Contains functionality for creating PyTorch DataLoaders for
the Stellators dataset.
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from StellatorsDataSet import StellatorsDataSet
from .utils import norm


def create_dataloaders(
    dataset: Dataset,
    train_size: float,
    batch_size: int, 
    num_workers: int=0
):
    """
    Create train and test data loaders for a given dataset.

    Args:
        dataset (Dataset): The dataset to be split into train and test sets.
        train_size (float): The proportion of the dataset to be used for training. It should be a float between 0 and 1.
        batch_size (int): The number of samples per batch.
        num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to NUM_WORKERS.

    Returns:
        train_loader (DataLoader): The data loader for the training set.
        test_loader (DataLoader): The data loader for the test set.
    """
    if train_size < 0 or train_size > 1:
        raise ValueError("train_size must be a float between 0 and 1.")
    # Define sizes for train and test datasets
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Normalize the data
    # Get the mean and standard deviation of the training dataset
    # We use the mean and standard deviation of the training dataset
    # to normalize the test dataset as well to avoid data leakage
    mean = torch.mean(torch.tensor(train_dataset.dataset.features), dim=0, dtype=torch.float32)
    mean_labels = torch.mean(torch.tensor(train_dataset.dataset.labels), dim=0, dtype=torch.float32)
    std = torch.std(torch.tensor(train_dataset.dataset.features), dim=0).float()
    std_labels = torch.std(torch.tensor(train_dataset.dataset.labels), dim=0).float()

    # Find Max and Min values
    max = torch.max(torch.tensor(train_dataset.dataset.features), dim=0).values.float()
    max_labels = torch.max(torch.tensor(train_dataset.dataset.labels), dim=0).values.float()
    min = torch.min(torch.tensor(train_dataset.dataset.features), dim=0).values.float()
    min_labels = torch.min(torch.tensor(train_dataset.dataset.labels), dim=0).values.float()

    train_dataset.dataset.mean = mean
    train_dataset.dataset.mean_labels = mean_labels
    train_dataset.dataset.std = std
    train_dataset.dataset.std_labels = std_labels
    train_dataset.dataset.max = max
    train_dataset.dataset.max_labels = max_labels
    train_dataset.dataset.min = min
    train_dataset.dataset.min_labels = min_labels
    test_dataset.dataset.mean = mean
    test_dataset.dataset.mean_labels = mean_labels
    test_dataset.dataset.std = std
    test_dataset.dataset.std_labels = std_labels
    test_dataset.dataset.max = max
    test_dataset.dataset.max_labels = max
    test_dataset.dataset.min = min
    test_dataset.dataset.min_labels = min

    # Preprocess the data
    train_dataset.dataset.transform = norm
    test_dataset.dataset.transform = norm

    # Turn datasets into iterable objects (batches)
    train_loader = DataLoader(dataset=train_dataset, # dataset to turn into iterable batches
                            batch_size=batch_size, # samples per batch
                            shuffle=False, # shuffle data (for training set only)
                            num_workers=num_workers, # subprocesses to use for data loading
                            # pin_memory=True # CUDA only
    ) 

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True
    )

    return train_loader, test_loader
