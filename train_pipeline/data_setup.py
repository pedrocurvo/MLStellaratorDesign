"""
Contains functionality for creating PyTorch DataLoaders for
the Stellators dataset.
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from StellaratorsDataSet import StellaratorDataSet
from .utils import norm, set_dataset_statistics


def create_dataloaders(
    dataset: Dataset,
    train_size: float,
    val_size: float,
    batch_size: int, 
    num_workers: int=0
):
    """
    Create train and test data loaders for a given dataset.

    Args:
        dataset (Dataset): The dataset to be split into train and test sets.
        train_size (float): The proportion of the dataset to be used for training. It should be a float between 0 and 1.
        val_size (float): The proportion of the dataset to be used for validation. It should be a float between 0 and 1.
        batch_size (int): The number of samples per batch.
        num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to NUM_WORKERS.

    Returns:
        train_loader (DataLoader): The data loader for the training set.
        val_loader (DataLoader): The data loader for the validation set.
        test_loader (DataLoader): The data loader for the test set.
        mean_std (dict): A dictionary containing the mean and standard deviation of the training dataset.
    """
    if train_size < 0 or train_size > 1 or val_size < 0 or val_size > 1:
        raise ValueError("train_size must be a float between 0 and 1.")
    if train_size + val_size > 1:
        raise ValueError("train_size + val_size must be less than 1.")
    # Define sizes for train and test datasets
    train_size = int(train_size * len(dataset))
    val_size = int(val_size * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

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

    # Set the mean and standard deviation of the training dataset
    set_dataset_statistics(train_dataset.dataset, mean, mean_labels, std, std_labels, max, max_labels, min, min_labels)
    set_dataset_statistics(val_dataset.dataset, mean, mean_labels, std, std_labels, max, max_labels, min, min_labels)
    set_dataset_statistics(test_dataset.dataset, mean, mean_labels, std, std_labels, max, max_labels, min, min_labels)

    # Preprocess the data
    train_dataset.dataset.transform = norm
    val_dataset.dataset.transform = norm
    test_dataset.dataset.transform = norm

    # Turn datasets into iterable objects (batches)
    train_loader = DataLoader(dataset=train_dataset, # dataset to turn into iterable batches
                            batch_size=batch_size, # samples per batch
                            shuffle=False, # shuffle data (for training set only)
                            num_workers=num_workers, # subprocesses to use for data loading
                            drop_last=True, # drop the last batch if it is not complete
                            # pin_memory=True # CUDA only
    ) 

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True
    )

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            # pin_memory=True
    )

    # Dictionary containing the mean and standard deviation of the training dataset
    mean_std = {"mean": mean, "std": std, "mean_labels": mean_labels, "std_labels": std_labels}

    return train_loader, val_loader, test_loader, mean_std
