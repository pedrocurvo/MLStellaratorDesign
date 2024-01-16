"""
Contains functionality for creating PyTorch DataLoaders for
the Stellators dataset.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from StellatorsDataSet import StellatorsDataSet

NUM_WORKERS = 0

def create_dataloaders(
    dataset: Dataset,
    train_size: int,
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
    # Define sizes for train and test datasets
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Get the mean and standard deviation of the training dataset
    mean = 0
    std = 0
    for X, y in train_dataset:
        mean += X
        std += X.std()
    mean /= len(train_dataset)
    std /= len(train_dataset)
    print(f"Mean: {mean}")
    print(f"Std: {std}")

    for X, y in train_dataset:
        print(X)
        break

    def try_norm(X, mean, std):
        return (X - mean) / std
    train_dataset.dataset.mean = mean
    train_dataset.dataset.std = std
    test_dataset.dataset.mean = mean
    test_dataset.dataset.std = std

    # Preprocess the data
    train_dataset.dataset.transform = try_norm
    test_dataset.dataset.transform = try_norm

    for X, y in train_dataset:
        print(X)
        break


    
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
