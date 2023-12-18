import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt

class StellatorsDataSet(Dataset):
    def __init__(self, npy_file, transform=None, normalization=None, sample_size=None):
        # Load data from the .npy file
        if sample_size:
            self.data = np.load(npy_file)[:sample_size, :]
        else:
            self.data = np.load(npy_file)
        self.transform = transform
        self.normalization = normalization
        self.features = self.data[:, 10:]
        self.labels = self.data[:, :10]
        # Dictionary of features and labels
        # Maps the feature/label name to the index in the data array
        self.data_dict = {'rc1' : 0,
                            'rc2' : 1,
                            'rc3' : 2,
                            'zs1' : 3,
                            'zs2' : 4,
                            'zs3' : 5,
                            'nfp' : 6,
                            'etabar' : 7,
                            'B2c' : 8,
                            'p2' : 9,
                            'iota' : 10,
                            'max_elongation' : 11,
                            'min_L_grad_B' : 12,
                            'min_R0' : 13,
                            'r_singularity' : 14,
                            'L_grad_grad_B' : 15,
                            'B20_variation' : 16,
                            'beta' : 17,
                            'DMerc_times_r2' : 18}
        self.features_dict = {'iota' : 0,
                              'max_elongation' : 1,
                              'min_L_grad_B' : 2,
                              'min_R0' : 3,
                              'r_singularity' : 4,
                              'L_grad_grad_B' : 5,
                              'B20_variation' : 6,
                              'beta' : 7,
                              'DMerc_times_r2' : 8}
        self.labels_dict = {'rc1' : 0,
                            'rc2' : 1,
                            'rc3' : 2,
                            'zs1' : 3,
                            'zs2' : 4,
                            'zs3' : 5,
                            'nfp' : 6,
                            'etabar' : 7,
                            'B2c' : 8,
                            'p2' : 9}
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns one sample of the data, data and label (X, y).
        features = torch.tensor(self.data[idx, 10:], dtype=torch.float32)
        labels = torch.tensor(self.data[idx, :10], dtype=torch.float32)

        return features, labels
    
    def view_distributions(self, variable=None, percentage=100, filename=None, show=True):
        # Visualize feature distributions using a percentage of the data
        subset_size = int((percentage / 100) * len(self.data))
        #subset_features = pd.DataFrame(self.features[:subset_size, :], columns=[f'col{i+1}' for i in range(self.features.shape[1])])
        subset_features = pd.DataFrame(self.data[:subset_size, self.data_dict[variable]], columns=[variable])
        # Replace inf values with NaN
        # subset_features = subset_features.replace([np.inf, -np.inf, np.nan], 0)
        plt.figure(figsize=(15, 10))
        sns.set(style="whitegrid")
        try:
            sns.histplot(subset_features[variable], kde=True)
        except:
            sns.histplot(subset_features[variable])
        plt.title(f"{variable.capitalize()} Distribution ({percentage}% of Data)")
        # Plot the distributions of the features
        # for i, column in enumerate(subset_features.columns):
        #     plt.subplot(3, 3, i + 1)
        #     sns.histplot(subset_features[column], kde=True)
        #     plt.title(column)

        plt.tight_layout()

        # Save figure if filename is specified
        if filename:
            plt.savefig(filename)

        if show:
            plt.show()
    
    def view_correlations(self, percentage=100, method='pearson', filename=None, show=True, variables='features'):
        # View correlations using a percentage of the data
        subset_size = int((percentage / 100) * len(self.data))
        if variables == 'features':
            subset_features = pd.DataFrame(self.features[:subset_size, :], columns=[key for key in self.features_dict.keys()])
        elif variables == 'labels':
            subset_features = pd.DataFrame(self.labels[:subset_size, :], columns=[key for key in self.labels_dict.keys()])

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
