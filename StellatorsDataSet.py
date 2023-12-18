import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt

class StellatorsDataSet(Dataset):
    def __init__(self, npy_file, transform=None, normalization=None):
        # Load data from the .npy file
        self.data = np.load(npy_file)
        self.transform = transform
        self.normalization = normalization
        self.features = self.data[:, 10:]
        self.labels = self.data[:, :10]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns one sample of the data, data and label (X, y).
        features = torch.tensor(self.data[idx, 10:], dtype=torch.float32)
        labels = torch.tensor(self.data[idx, :10], dtype=torch.float32)

        return features, labels
    
    def view_distributions(self, percentage=100, filename=None, show=True):
        # Visualize feature distributions using a percentage of the data
        subset_size = int((percentage / 100) * len(self.data))
        subset_features = pd.DataFrame(self.features[:subset_size, :], columns=[f'col{i+1}' for i in range(self.features.shape[1])])

        plt.figure(figsize=(15, 10))
        sns.set(style="whitegrid")

        for i, column in enumerate(subset_features.columns):
            plt.subplot(3, 3, i + 1)
            sns.histplot(subset_features[column], kde=True)
            plt.title(column)

        plt.tight_layout()

        # Save figure if filename is specified
        if filename:
            plt.savefig(filename)

        if show:
            plt.show()
    
    def view_correlations(self, percentage=100, method='pearson', filename=None, show=True):
        # View correlations using a percentage of the data
        subset_size = int((percentage / 100) * len(self.data))
        subset_features = pd.DataFrame(self.features[:subset_size, :], columns=[f'col{i+1}' for i in range(self.features.shape[1])])

        plt.figure(figsize=(12, 18))
        sns.set(style="white")

        # Compute the correlation matrix for the inputs 
        correlation_matrix = subset_features.corr(method=method)

        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Feature Correlations (Method: {method.capitalize()}, {percentage}% of Data)")

        # Save figure if filename is specified
        if filename:
            plt.savefig(filename)

        if show:
            plt.show()
