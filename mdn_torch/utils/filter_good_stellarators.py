import pandas as pd
import numpy as np

def filter_good_stellarators(input_file: str = None, output_file: str = None):
    """
    Filter the dataset of stellarators to select only the good stellarators.
    The conditions for a good stellarator are:
    - axis_length > 0
    - |iota| >= 0.2
    - max_elongation <= 10
    - |min_L_grad_B| >= 0.1
    - |min_R0| >= 0.3
    - r_singularity >= 0.05
    - |L_grad_grad_B| >= 0.1
    - B20_variation <= 5
    - beta >= 1e-4
    - DMerc_times_r2 > 0

    Parameters
    ----------
    input_file : str
        Path to the input CSV file containing the dataset of stellarators.
    output_file : str
        Path to the output CSV file where the filtered dataset will be saved.
    """
    # Load the dataset
    df = pd.read_csv(input_file)

    # Print the shape of the dataset
    print(f"Shape of the dataset: {df.shape}")

    # Conditions for a good stellarator
    condition = (
        (df['axis_length'] > 0) &
        (np.fabs(df['iota']) >= 0.2) &
        (df['max_elongation'] <= 10) &
        (np.fabs(df['min_L_grad_B']) >= 0.1) &
        (np.fabs(df['min_R0']) >= 0.3) &
        (df['r_singularity'] >= 0.05) &
        (np.fabs(df['L_grad_grad_B']) >= 0.1) &
        (df['B20_variation'] <= 5) &
        (df['beta'] >= 1e-4) &
        (df['DMerc_times_r2'] > 0)
    )

    # Select rows that satisfy the combined condition
    df = df[condition]

    # Display the first few rows of the filtered dataset
    print(df.head())

    # Print the shape of the filtered dataset
    print(f"Shape of the filtered dataset: {df.shape}")

    # Save the filtered dataset to a new CSV file
    df.to_csv(output_file, index=False)



