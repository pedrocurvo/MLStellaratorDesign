import numpy as np

def sample_output(dataset):
    """
    Sample a random values from random rows of the dataset to generate a new sample to the MDN Neural Network.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset containing the stellarator data.
    
    Returns
    -------
    list
        A sample of output values for the Qsc (axis_length, iota, max_elongation, min_L_grad_B, min_R0, r_singularity,
        L_grad_grad_B, B20_variation, beta, DMerc_times_r2).
    """



    cols = ['axis_length', 'iota', 'max_elongation',
            'min_L_grad_B', 'min_R0', 'r_singularity',
            'L_grad_grad_B', 'B20_variation', 'beta',
            'DMerc_times_r2']

    n = dataset.shape[0]

    row_index = np.random.randint(n)  # Randomly select a row index

    sample = [dataset[col].iloc[np.random.randint(n)] if col != 'r_singularity' and col != 'beta'
              else dataset[col].iloc[row_index] for col in cols]

    return sample