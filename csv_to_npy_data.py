from tqdm import tqdm # for progress bar visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qsc import Qsc
import time 
from jax import grad
from StellatorsDataSet import StellatorsDataSet


# Convert .csv file to .npy file by chuncks
# This is useful for large datasets that cannot be loaded into memory all at once
# Path to your CSV file
csv_file_path = 'data/dataset.csv'

# Path to your output .npy file
npy_file_path = 'data/dataset.npy'

# Chunk size (adjust based on your available memory)
chunk_size = 10000

# NP array to store the data
data = np.empty((0, 19))

# Iterate over chunks of the CSV file
for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
    # Assuming the chunk is a DataFrame, convert it to a NumPy array
    data = np.concatenate((data, chunk.to_numpy()), axis=0)

# Print the shape of the data
print(data.shape)

# Save the data as a NumPy array
np.save(npy_file_path, data)
