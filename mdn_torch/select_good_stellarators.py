import pandas as pd 
import numpy as np

# Load the dataset
df = pd.read_csv('../data/fourth_dataset.csv')

# Print the shape of the dataset
print(f"Shape of the dataset: {df.shape}")

# Select the rows with axis_length > 0
df = df[df['axis_length'] > 0]

# Select the rows with |iota| >= 0.2
df = df[np.fabs(df['iota']) >= 0.2]

# Select the rows with max_elongation <= 10
df = df[df['max_elongation'] <= 10]

# Select the rows with |min_L_grad_B| >= 0.1
df = df[np.fabs(df['min_L_grad_B']) >= 0.1]

# Select the rows with |min_R0| >= 0.3
df = df[np.fabs(df['min_R0']) >= 0.3]

# Select the rows with r_singularity >= 0.05
df = df[df['r_singularity'] >= 0.05]

# Select the rows with |L_grad_grad_B| >= 0.1
df = df[np.fabs(df['L_grad_grad_B']) >= 0.1]

# Select the rows with B20_variation <= 5
df = df[df['B20_variation'] <= 5]

# Select the rows with beta >= 1e-4
df = df[df['beta'] >= 1e-4]

# Select the rows with DMerc_times_r2 > 0
df = df[df['DMerc_times_r2'] > 0]

# Display the first few rows of the filtered dataset
print(df.head())

# Print the shape of the filtered dataset
print(f"Shape of the filtered dataset: {df.shape}")

# Save the filtered dataset to a new CSV file
df.to_csv('./data_good/fourth_good_stels.csv', index=False)


