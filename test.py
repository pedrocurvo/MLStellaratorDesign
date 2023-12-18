from StellatorsDataSet import StellatorsDataSet
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import optim

# Load the dataset
dataset = StellatorsDataSet('data/dataset.npy', sample_size=1000)  
print(f'Number of samples: {len(dataset)}')
print(f'Number of features: {dataset.features.shape[1]}')
print(f'Number of labels: {dataset.labels.shape[1]}')

dataset.view_distributions(percentage=100, variable='beta', show=True)
dataset.view_correlations(percentage=100, show=True, method='spearman', variables='features')
dataset.view_correlations(percentage=100, show=True, method='pearson', variables='features')

