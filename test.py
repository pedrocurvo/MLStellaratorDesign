from StellatorsDataSet import StellatorsDataSet
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import numpy as np
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = StellatorsDataSet('data/dataset.npy')  
print(f'Number of samples: {len(dataset)}')
print(f'Number of features: {dataset.features.shape[1]}')
print(f'Number of labels: {dataset.labels.shape[1]}')

dataset.view_distributions(percentage=100, variables=['nfp', 'iota'], show=True, overlap=False)
dataset.view_distributions(percentage=100, variables=['nfp', 'iota'], show=True, overlap=True)

dataset.view_correlations(percentage=100, show=True, method='spearman', variables='features')
dataset.view_correlations(percentage=100, show=True, method='pearson', variables='features')
dataset.view_correlations(percentage=100, show=True, method='kendall', variables='features')

dataset.calculate_data_counts(IOTA_MIN = 0.2,
                              MAX_ELONGATION = 10,
                              MIN_MIN_L_GRAD_B = 0.1,
                              MIN_MIN_R0 = 0.3,
                              MIN_R_SINGULARITY = 0.05,
                              MIN_L_GRAD_GRAD_B = 0.01,
                              MAX_B20_VARIATION = np.inf,
                              MIN_BETA = 1e-4,
                              MIN_DMERC_TIMES_R2 = 0)



