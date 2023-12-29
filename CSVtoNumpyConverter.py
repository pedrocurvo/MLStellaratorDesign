import argparse
import numpy as np
import pandas as pd

# Convert .csv file to .npy file by chuncks
# This is useful for large datasets that cannot be loaded into memory all at once

# Set up the argument parser
parser = argparse.ArgumentParser(description='Convert CSV file to NPY format.')
parser.add_argument('csv_file_path', type=str, help='Path to your CSV file')

# Parse the arguments
args = parser.parse_args()

# Path to your CSV file (from command-line argument)
csv_file_path = args.csv_file_path

# Automatically set the output .npy file path
npy_file_path = csv_file_path.replace('.csv', '.npy')

# Chunk size (adjust based on your available memory)
chunk_size = 10000

# NP array to store the data
data = np.empty((0, 19))

# Iterate over chunks of the CSV file
for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
    # Assuming the chunk is a DataFrame, convert it to a NumPy array
    data = np.concatenate((data, chunk.to_numpy()), axis=0)

# Print the shape of the data
print(f"\nThe data has been successfully processed and has been saved to {npy_file_path}")
print(f"Shape of the data array: {data.shape}\n")

# Save the data as a NumPy array
np.save(npy_file_path, data)
