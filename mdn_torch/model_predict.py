import torch
import signal
import multiprocessing
import torch_optimizer as optim
from tqdm import tqdm

# Add Parent Directory to Python Path
# Inside your Python script within the external_package directory
import sys
import os
import numpy as np

# Get the parent directory of the current directory (external_package)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_dir)
print(parent_dir)

from StellaratorDataSet import StellaratorDataSetInverse
# Measure time
from timeit import default_timer as timer
from datetime import datetime
from train_pipeline import engine, utils, data_setup
from MDNFullCovariance import MDNFullCovariance

# -----------------------------------------------------------------------------

def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

cpus = os.cpu_count()

# -----------------------------------------------------------------------------

# Important for num_workers > 0
if __name__ == "__main__":
    # Dataset
    # Load the data
    full_dataset = StellaratorDataSetInverse(npy_file='../data/dataset.npy')

    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    # elif torch.backends.mps.is_available():
    #     device = "mps" # Apple GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"Using device: {device}")

    # Setup the Hyperparameters
    BATCH_SIZE = 500
    NUM_OF_WORKERS = os.cpu_count()

    # Turn datasets into iterable objects (batches)
    # Create DataLoaders with help from data_setup.py
    train_dataloader, val_dataloader, test_dataloader, mean_std = data_setup.create_dataloaders(dataset=full_dataset,
                                                                    train_size=0.5,
                                                                    val_size=0.2,
                                                                    batch_size=BATCH_SIZE,
                                                                    num_workers=NUM_OF_WORKERS
)



    # Create model
    model = MDNFullCovariance(input_dim=10,
                            output_dim=10,
                            num_gaussians=62
    ).to(device)
    
    # Load a previous model (optional: uncomment if you want to load a previous model): transfer learning
    model.load_state_dict(torch.load("models/MDNFullCovariance/2024_03_28_11_53_42.pth"))

    # Sample from .csv 
    import pandas as pd
    from preditcions_utils import sample_output, check_criteria, run_qsc, round_nfp

    df = pd.read_csv('../data/good_stellarators_dataset_first.csv')


    fname = '../data/predict_first.csv'
    print('Writing:', fname)

    if os.path.exists(fname):
        f = open(fname, 'a')
    else:
        f = open(fname, 'w')
    print(','.join(df.columns), file=f)

    it = 0
    progress_bar = tqdm(
        range(6000000), 
        desc=f"Predicting", 
        leave=False,
        disable=False,
        colour="green"
    )

    for i in progress_bar:
        sample = sample_output(df)

        with torch.no_grad():
            # Transform into tensor
            sample = torch.tensor(sample).float().to(device).unsqueeze(0)

            # Remove mean and divide by std
            sample = (sample - mean_std["mean"].to(device)) / mean_std["std"].to(device)

            # Pass through model
            output = model.getMixturesSample(sample, device)

            # Add mean and multiply by std
            output = output * mean_std["std_labels"].to(device) + mean_std["mean_labels"].to(device)

            # Round the 7th element of the output
            output[0][6] = torch.round(output[0][6])

            # Run qsc
            output = output.cpu().numpy()

            try: 
                output[0] = round_nfp(output[0])
                qsc_values = run_qsc(output[0])

                # Check criteria
                if check_criteria(qsc_values):
                    it += 1

                assert not np.isnan(qsc_values).any()
                assert not np.isinf(qsc_values).any()
                assert not (np.fabs(qsc_values) > np.finfo(np.float32).max).any()

                values = np.concatenate([output[0], qsc_values], dtype=str)
                print(','.join(values), file=f)

            except Warning:
                continue

            except AssertionError:
                continue

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "gstel": it,
                    "%": it / (i + 1) * 100,
                }
            )
            progress_bar.update()

            
            
        