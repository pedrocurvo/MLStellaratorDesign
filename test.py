from StellatorsDataSet import StellatorsDataSet
import numpy as np


# -----------------------------------------------------------------------------
# Load the dataset
dataset = StellatorsDataSet('data/dataset.npy')  
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
                              MIN_L_GRAD_GRAD_B = 0.01,
                              MAX_B20_VARIATION = np.inf,
                              MIN_BETA = 1e-4,
                              MIN_DMERC_TIMES_R2 = 0,
                              return_object=True)

# -----------------------------------------------------------------------------
# View the distributions of the dataset with restrictions applied

dataset_with_restrictions.view_distributions(percentage=100, variables=['nfp', 'iota'], show=True, overlap=False)



