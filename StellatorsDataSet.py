import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

class StellatorsDataSet(Dataset):
    def __init__(self, npy_file, transform=None, normalization=None, sample_size=None):
        # If npy_file is a npy array
        if isinstance(npy_file, np.ndarray):
            self.data = npy_file
        elif npy_file.endswith('.npy'):
            self.data = np.load(npy_file)
        elif npy_file.endswith('.csv'):
            self.data = pd.read_csv(npy_file).to_numpy()
        # If npy_file is a path to a npy file
        else:
            raise TypeError("npy_file should be a path to a npy file or a npy array")
        
        # If sample_size is specified, only use the first sample_size rows
        if sample_size:
            self.data = self.data[:sample_size, :]
        
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


    def view_distributions(self, variables=None, percentage=100, filename=None, show=True, overlap=False):
        """
        Plots the distribution of the specified variables from the dataset.

        Args:
            variables (str, list): The variable(s) to plot. Should be a string or a list of strings. 
                If 'features', all features will be plotted. If 'labels', all labels will be plotted.
            percentage (int, optional): The percentage of the data to include in the plot. Defaults to 100.
            filename (str, optional): If specified, the plot will be saved as this filename. Defaults to None.
            show (bool, optional): If True, the plot will be displayed. Defaults to True.
            overlap (bool, optional): If True, the distributions of all variables will be plotted on the same graph. Defaults to False.

        Raises:
            ValueError: If no variables are provided or if the percentage is not between 0 and 100.
            TypeError: If variables is not a string or a list of strings.

        Returns:
            None
        """
        if variables == 'features':
            variables = [key for key in self.features_dict.keys()]
        elif variables == 'labels':
            variables = [key for key in self.labels_dict.keys()]
        # Validate inputs
        if not variables:
            raise ValueError("No variables provided")
        if not isinstance(variables, (list, str)):
            raise TypeError("Variables should be a string or a list of strings")
        if percentage <= 0 or percentage > 100:
            raise ValueError("Percentage must be between 0 and 100")

        # Convert to list if variables is a string
        variables = [variables] if isinstance(variables, str) else variables

        # Calculate the subset size
        subset_size = int((percentage / 100) * len(self.data))

        # Set plot style
        sns.set(style="whitegrid")

        # Plot the distribution of the variable
        if overlap:
            plt.figure(figsize=(15, 10))
            for variable in variables:
                sns.histplot(self.data[:subset_size, self.data_dict[variable]], kde=True)
            # Set the title of the plot
            title = f"{' , '.join(variables).capitalize()} Distribution ({percentage}% of Data)"
            plt.title(title)
            plt.tight_layout()

            # Save figure if filename is specified
            if filename:
                plt.savefig(filename)
            # Show the plot if show is True
            if show:
                plt.show()
        else:
            for variable in variables:
                plt.figure(figsize=(15, 10))
                # Plot the distribution of the variable with a kde if continuous
                sns.histplot(self.data[:subset_size, self.data_dict[variable]], kde=True, bins=10)
                # Set the title of the plot
                plt.title(f"{variable.capitalize()} Distribution ({percentage}% of Data)")
                plt.tight_layout()

                # Save figure if filename is specified
                if filename:
                    plt.savefig(filename)
                # Show the plot if show is True
                if show:
                    plt.show()
    

    def view_correlations(self, percentage=100, method='pearson', filename=None, show=True, variables='features'):
        """
        Plots a heatmap of the correlations between the specified variables from the dataset.

        Args:
            percentage (int, optional): The percentage of the data to include in the plot. Defaults to 100.
            method (str, optional): The method to use for computing correlations. Defaults to 'pearson'.
            filename (str, optional): If specified, the plot will be saved as this filename. Defaults to None.
            show (bool, optional): If True, the plot will be displayed. Defaults to True.
            variables (str, optional): The variables to include in the correlation matrix. Can be 'features', 'labels', or 'all'. Defaults to 'features'.

        Raises:
            ValueError: If the percentage is not between 0 and 100, or if the method is not one of 'pearson', 'kendall', 'spearman'.
            TypeError: If variables is not a string.

        Returns:
            None
        """
            # function implementation...
        # View correlations using a percentage of the data
        subset_size = int((percentage / 100) * len(self.data))
        if variables == 'features':
            subset = pd.DataFrame(self.features[:subset_size, :], columns=[key for key in self.features_dict.keys()])
        elif variables == 'labels':
            subset = pd.DataFrame(self.labels[:subset_size, :], columns=[key for key in self.labels_dict.keys()])
        elif variables == 'all':
            subset = pd.DataFrame(self.data[:subset_size, :], columns=[key for key in self.data_dict.keys()])

        plt.figure(figsize=(12, 18))
        sns.set(style="white")

        # Compute the correlation matrix for the inputs 
        correlation_matrix = subset.corr(method=method)

        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"{variables.capitalize()} Correlations (Method: {method.capitalize()}, {percentage}% of Data)")
        # Tilt the x-axis labels
        plt.xticks(rotation=35)
        plt.yticks(rotation=35)

        # Save figure if filename is specified
        if filename:
            plt.savefig(filename)

        if show:
            plt.show()


    def calculate_data_counts(self, IOTA_MIN, MAX_ELONGATION, MIN_MIN_L_GRAD_B, MIN_MIN_R0, MIN_R_SINGULARITY, MIN_L_GRAD_GRAD_B, MAX_B20_VARIATION, MIN_BETA, MIN_DMERC_TIMES_R2, return_object=False):
        file = self.data
        # 9 features to check 
        iota = np.count_nonzero(np.fabs(file[:, 10]) >= IOTA_MIN)
        max_elongation = np.count_nonzero(file[:, 11] <= MAX_ELONGATION)
        min_L_grad_B = np.count_nonzero(np.fabs(file[:, 12]) >= MIN_MIN_L_GRAD_B)
        min_R0 = np.count_nonzero(file[:, 13] >= MIN_MIN_R0)
        r_singularity = np.count_nonzero(file[:, 14] >= MIN_R_SINGULARITY)
        L_grad_grad_B = np.count_nonzero(np.fabs(file[:, 15]) >= MIN_L_GRAD_GRAD_B)
        B20_variation = np.count_nonzero(file[:, 16] <= MAX_B20_VARIATION)
        beta = np.count_nonzero(file[:, 17] >= MIN_BETA)
        DMerc_times_r2 = np.count_nonzero(file[:, 18] > MIN_DMERC_TIMES_R2)

        # Percentage of data
        per_iota = iota/len(file) * 100
        per_max_elongation = max_elongation/len(file) * 100
        per_min_L_grad_B = min_L_grad_B/len(file) * 100
        per_min_R0 = min_R0/len(file) * 100
        per_r_singularity = r_singularity/len(file) * 100
        per_L_grad_grad_B = L_grad_grad_B/len(file) * 100
        per_B20_variation = B20_variation/len(file) * 100
        per_beta = beta/len(file) * 100
        per_DMerc_times_r2 = DMerc_times_r2/len(file) * 100

        # Define the data as a list of lists
        data = [
            [f'abs(iota) > {IOTA_MIN}', iota, round(per_iota, 2)],
            [f'max_elongation <= {MAX_ELONGATION}', max_elongation, round(per_max_elongation, 2)],
            [f'abs(min_L_grad_B) >= {MIN_MIN_L_GRAD_B}', min_L_grad_B, round(per_min_L_grad_B, 2)],
            [f'min_R0 >= {MIN_MIN_R0}', min_R0, round(per_min_R0, 2)],
            [f'r_singularity >= {MIN_R_SINGULARITY}', r_singularity, round(per_r_singularity, 2)],
            [f'abs(L_grad_grad_B) >= {MIN_L_GRAD_GRAD_B}', L_grad_grad_B, round(per_L_grad_grad_B, 2)],
            [f'B20_variation <= {MAX_B20_VARIATION}', B20_variation, round(per_B20_variation, 2)],
            [f'beta >= {MIN_BETA}', beta, round(per_beta, 2)],
            [f'DMerc_times_r2 > {MIN_DMERC_TIMES_R2}', DMerc_times_r2, round(per_DMerc_times_r2, 2)],
        ]

        # Print the table
        print(tabulate(data, headers=['Feature', 'Counts', 'Per of Data'], tablefmt='grid'))

        # Count all the data with all restrictions
        count = np.count_nonzero((np.fabs(file[:, 10]) >= IOTA_MIN) &
                                (file[:, 11] <= MAX_ELONGATION) &
                                (np.fabs(file[:, 12]) >= MIN_MIN_L_GRAD_B) &
                                (file[:, 13] >= MIN_MIN_R0) &
                                (file[:, 14] >= MIN_R_SINGULARITY) &
                                (np.fabs(file[:, 15]) >= MIN_L_GRAD_GRAD_B) &
                                (file[:, 16] <= MAX_B20_VARIATION)
                                & (file[:, 17] >= MIN_BETA)
                                & (file[:, 18] > MIN_DMERC_TIMES_R2))

        # Percentage of data
        per_count = count/len(file) * 100

        # Define the data as a list of lists
        data = [
            ['All Restrictions', count, per_count],
        ]

        # Print the table
        print(tabulate(data, headers=['Feature', 'Counts', 'Per of Data'], tablefmt='grid'))

        # Return an object that respects all the restrictions
        if return_object:
            ##### Need to specify a deep copy of the data and then replace self.data and then return
            return StellatorsDataSet(file[(np.fabs(file[:, 10]) >= IOTA_MIN) &
                        (file[:, 11] <= MAX_ELONGATION) &
                        (np.fabs(file[:, 12]) >= MIN_MIN_L_GRAD_B) &
                        (file[:, 13] >= MIN_MIN_R0) &
                        (file[:, 14] >= MIN_R_SINGULARITY) &
                        (np.fabs(file[:, 15]) >= MIN_L_GRAD_GRAD_B) &
                        (file[:, 16] <= MAX_B20_VARIATION)
                        & (file[:, 17] >= MIN_BETA)
                        & (file[:, 18] > MIN_DMERC_TIMES_R2)],
                        transform=self.transform,
                        normalization=self.normalization)
