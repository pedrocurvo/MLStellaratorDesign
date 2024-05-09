import pandas as pd
import glob

import pandas as pd
import os

def merge_csvs(csv_files: list = None, output_file: str = None):
    """
    Combine multiple CSV files into a single CSV file.

    Parameters
    ----------
    csv_files : list
        List of paths to the input CSV files.
    output_file : str
        Path to the output CSV file where the combined data will be saved.
    
    Example
    -------
    csv_files = ['first_good_stels.csv', 'second_good_stels.csv', 'third_good_stels.csv',
                 'fourth_good_stels.csv', 'fifth_good_stels.csv', 'sixth_good_stels.csv']
    combine_csv_files(csv_files)
    """
    # Initialize an empty list to store file paths
    file_paths = []

    # Iterate through each CSV file and store their paths
    for csv_file in csv_files:
        file_paths.append(csv_file)

    # Concatenate all CSV files into a single dataframe
    combined_df = pd.concat((pd.read_csv(file, header=0) for file in file_paths), ignore_index=True)

    # Write the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)


