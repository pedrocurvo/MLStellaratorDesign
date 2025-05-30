import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from qsc import Qsc

class StellaratorDataSetInverse(Dataset):
    def __init__(self,
                 npy_file,
                 transform=None,
                 normalization=None,
                 sample_size: int = None,
                 features_names: list[str] = ['axis_length', 'iota', 'max_elongation', 'min_L_grad_B', 'min_R0', 
                    'r_singularity', 'L_grad_grad_B', 'B20_variation', 'beta', 'DMerc_times_r2'],
                 labels_names: list[str] = ['rc1', 'rc2', 'rc3', 'zs1', 'zs2', 'zs3', 'nfp', 'etabar', 'B2c', 'p2'],
                 dtype=torch.float32):
        
        # Keep the name of the features and labels
        self.features_names = features_names
        self.labels_names = labels_names
        self.separation_idx = len(labels_names)

        # Load the data
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
        self.features = self.data[:, self.separation_idx:]
        self.labels = self.data[:, :self.separation_idx]
        # ---------------------------------------------------------------------
        # The Atrributes below are used for normalization
        # ---------------------------------------------------------------------
        # Mean and standard deviation of the data
        self.mean = None
        self.mean_labels = None
        self.std = None
        self.std_labels = None
        # Min and max of the data
        self.min = None
        self.min_labels = None
        self.max = None
        self.max_labels = None

        # Dictionary of features and labels
        # Maps the feature/label name to the index in the data array
        self.data_dict = {name: idx for idx, name in enumerate(labels_names + features_names)}
        self.features_dict = {name: idx for idx, name in enumerate(features_names)}
        self.labels_dict = {name: idx for idx, name in enumerate(labels_names)}
        
    #------------------------------------------------------------------------------
    # Essential for DataLoader to work
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns one sample of the data, data and label (X, y).
        features = torch.tensor(self.data[idx, self.separation_idx:], dtype=torch.float32)
        labels = torch.tensor(self.data[idx, :self.separation_idx], dtype=torch.float32)
        if self.transform:
            features = self.transform(features, self.mean, self.std)
            labels = self.transform(labels, self.mean_labels, self.std_labels)
        # Remove nfp from the label which is index 6 
        #labels = torch.cat((labels[:6], labels[7:]))
        return features, labels
    
    #------------------------------------------------------------------------------
    # Iterate over the dataset and return a Qsc object
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.data):
            try:
                # Unpack the result tuple for better readability
                rc1, rc2, rc3, zs1, zs2, zs3, nfp, etabar, B2c, p2 = self.data[self.index]
                
                # Create Qsc object
                qsc_object = Qsc(rc=[1., rc1, rc2, rc3], zs=[0., zs1, zs2, zs3], nfp=nfp, etabar=etabar, B2c=B2c, p2=p2, order='r2')
                
                self.index += 1
                return qsc_object
            except Exception as e:
                # Handle any exceptions that may occur during iteration
                print(f"An error occurred while creating Qsc object: {e}")
                self.index += 1
                return None
        else:
            # Signal the end of iteration
            raise StopIteration


    #------------------------------------------------------------------------------
    # View the distributions of the dataset
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
            variables = self.features_names
        elif variables == 'labels':
            variables = self.labels_names
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
        sns.set_theme(style="whitegrid")

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
    
    #------------------------------------------------------------------------------
    # View the correlations between the features and the labels
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
        sns.set_theme(style="white")

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

    #------------------------------------------------------------------------------
    def calculate_data_counts(self, AXIS_LENGTH, IOTA_MIN, MAX_ELONGATION, MIN_MIN_L_GRAD_B, MIN_MIN_R0, MIN_R_SINGULARITY, MIN_L_GRAD_GRAD_B, MAX_B20_VARIATION, MIN_BETA, MIN_DMERC_TIMES_R2, return_object=False):
        file = self.data
        # 9 features to check 
        axis_length = np.count_nonzero(file[:, self.data_dict['axis_length']] > AXIS_LENGTH)
        iota = np.count_nonzero(np.fabs(file[:, self.data_dict['iota']]) >= IOTA_MIN)
        max_elongation = np.count_nonzero(file[:, self.data_dict['max_elongation']] <= MAX_ELONGATION)
        min_L_grad_B = np.count_nonzero(np.fabs(file[:, self.data_dict['min_L_grad_B']]) >= MIN_MIN_L_GRAD_B)
        min_R0 = np.count_nonzero(file[:, self.data_dict['min_R0']] >= MIN_MIN_R0)
        r_singularity = np.count_nonzero(file[:, self.data_dict['r_singularity']] >= MIN_R_SINGULARITY)
        L_grad_grad_B = np.count_nonzero(np.fabs(file[:, self.data_dict['L_grad_grad_B']]) >= MIN_L_GRAD_GRAD_B)
        B20_variation = np.count_nonzero(file[:, self.data_dict['B20_variation']] <= MAX_B20_VARIATION)
        beta = np.count_nonzero(file[:, self.data_dict['beta']] >= MIN_BETA)
        DMerc_times_r2 = np.count_nonzero(file[:, self.data_dict['DMerc_times_r2']] > MIN_DMERC_TIMES_R2)

        # Percentage of data
        per_axis_length = axis_length/len(file) * 100
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
            [f'axis_length > {AXIS_LENGTH}', axis_length, round(per_axis_length, 2)],
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
        count = np.count_nonzero((file[:,  self.data_dict['axis_length']] > AXIS_LENGTH) &
                                (np.fabs(file[:, self.data_dict['iota']]) >= IOTA_MIN) &
                                (file[:, self.data_dict['max_elongation']] <= MAX_ELONGATION) &
                                (np.fabs(file[:, self.data_dict['min_L_grad_B']]) >= MIN_MIN_L_GRAD_B) &
                                (file[:, self.data_dict['min_R0']] >= MIN_MIN_R0) &
                                (file[:, self.data_dict['r_singularity']] >= MIN_R_SINGULARITY) &
                                (np.fabs(file[:, self.data_dict['L_grad_grad_B']]) >= MIN_L_GRAD_GRAD_B) &
                                (file[:, self.data_dict['B20_variation']] <= MAX_B20_VARIATION)
                                & (file[:, self.data_dict['beta']] >= MIN_BETA)
                                & (file[:, self.data_dict['DMerc_times_r2']] > MIN_DMERC_TIMES_R2))

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
            data = self.data[
                                (file[:,  self.data_dict['axis_length']] > AXIS_LENGTH) &
                                (np.fabs(file[:, self.data_dict['iota']]) >= IOTA_MIN) &
                                (file[:, self.data_dict['max_elongation']] <= MAX_ELONGATION) &
                                (np.fabs(file[:, self.data_dict['min_L_grad_B']]) >= MIN_MIN_L_GRAD_B) &
                                (file[:, self.data_dict['min_R0']] >= MIN_MIN_R0) &
                                (file[:, self.data_dict['r_singularity']] >= MIN_R_SINGULARITY) &
                                (np.fabs(file[:, self.data_dict['L_grad_grad_B']]) >= MIN_L_GRAD_GRAD_B) &
                                (file[:, self.data_dict['B20_variation']] <= MAX_B20_VARIATION)
                                & (file[:, self.data_dict['beta']] >= MIN_BETA)
                                & (file[:, self.data_dict['DMerc_times_r2']] > MIN_DMERC_TIMES_R2)
                        ]

            return StellaratorDataSetInverse(data,
                        transform=self.transform,
                        normalization=self.normalization)

    #------------------------------------------------------------------------------
    # Get the Qsc object from the dataset
    def getQSC(self, idx):
        try:
            # Extract the label corresponding to the given index
            label = self.labels[idx]
            
            # Extract individual values from the label
            rc_values = label[:3]
            zs_values = label[3:6]
            nfp_value = label[6]
            etabar_value = label[7]
            B2c_value = label[8]
            p2_value = label[9]
            
            # Create Qsc object
            stel = Qsc(rc=[1.] + rc_values, zs=[0.] + zs_values, nfp=nfp_value, etabar=etabar_value, B2c=B2c_value, p2=p2_value, order='r2')
            
            return stel
        except IndexError:
            # Handle index out of range error
            print(f"Index '{idx}' is out of range")
            return None
        except Exception as e:
            # Handle any other exceptions
            print(f"An error occurred: {e}")
            return None
