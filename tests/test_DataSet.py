import numpy as np
from qsc import Qsc
import os
import sys

# -----------------------------------------------------------------------------
# Get the directory of the script or notebook
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
# -----------------------------------------------------------------------------
from StellatorsDataSet import StellatorsDataSet

# -----------------------------------------------------------------------------
# Load the dataset
dataset = StellatorsDataSet('../data/dataset.npy')  
print(f'Number of samples: {len(dataset)}')
print(f'Number of features: {dataset.features.shape[1]}')
print(f'Number of labels: {dataset.labels.shape[1]}')

# -----------------------------------------------------------------------------
# View the distributions of the dataset

# dataset.view_distributions(percentage=100, variables=['nfp', 'iota'], show=True, overlap=False)
# dataset.view_distributions(percentage=100, variables=['nfp', 'iota'], show=True, overlap=True)

# -----------------------------------------------------------------------------
# See the correlations between the features and the labels

# dataset.view_correlations(percentage=100, show=True, method='spearman', variables='features')
# dataset.view_correlations(percentage=100, show=True, method='pearson', variables='features')
# dataset.view_correlations(percentage=100, show=True, method='kendall', variables='features')

# -----------------------------------------------------------------------------
# Count the number of samples respecting the restrictions and return a new StellatorsDataSet object with the restricted data

dataset_with_restrictions = dataset.calculate_data_counts(IOTA_MIN = 0.2,
                              MAX_ELONGATION = 10,
                              MIN_MIN_L_GRAD_B = 0.1,
                              MIN_MIN_R0 = 0.3,
                              MIN_R_SINGULARITY = 0.05,
                              MIN_L_GRAD_GRAD_B = 0.1,
                              MAX_B20_VARIATION = 5,
                              MIN_BETA = 1e-4,
                              MIN_DMERC_TIMES_R2 = 0,
                              return_object=True)

dataset_with_restrictions.view_correlations(method='spearman', variables='features', show=True)

# -----------------------------------------------------------------------------
# View the distributions of the dataset with restrictions applied

# dataset_with_restrictions.view_distributions(percentage=100, variables='labels', show=True, overlap=False)

# -----------------------------------------------------------------------------
for stel in dataset_with_restrictions:
    try:
        stel.plot_boundary()
    except:
        pass


