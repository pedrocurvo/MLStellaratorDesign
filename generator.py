'''
Script to generate new stellarators using a trained model. 
The script will generate new stellarators and check if they meet the criteria of good stellarators. 
The script will save the new stellarators in a file.
'''

import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from mdn_torch import utils
from mdn_torch import MDNFullCovariance
import argparse

# Define a argparser to run the script
def parser():
    """Parses arguments from the command line.

    Returns:
        argparse.Namespace: Arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Generate New Stellarators", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="mdn_torch/models/MDNFullCovariance/model_05.pth", help="Path to the model")
    parser.add_argument("--model_mean", type=str, default="mdn_torch/models/mean_std_05.pth", help="Path to the mean_std file with the mean and std of the model")
    parser.add_argument("--from_data", type=str, default="./data/5_dataset/fifth_good_stels.csv", help="Path to the data from which to generate new stellarators")
    parser.add_argument("--to_data", type=str, default="./new_dataset.csv", help="Path to the file to save the new stellarators")
    parser.add_argument("--num_samples", type=int, default=100000, help="Number of samples to generate")

    args = parser.parse_args()
    return args



# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Get the arguments from the command line
    args = parser()

    MODEL = args.model
    MODEL_MEAN = args.model_mean
    FROM_DATA = args.from_data
    TO_DATA = args.to_data
    NUM_SAMPLES = args.num_samples

    # -----------------------------------------------------------------------------
    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"Using device: {device}")
    
    # Load mean_std from a file
    mean_std = torch.load(MODEL_MEAN, map_location=torch.device('cpu'))

    # Create model
    model = MDNFullCovariance.MDNFullCovariance(input_dim=10,
                            output_dim=10,
                            num_gaussians=62
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL, map_location=torch.device('cpu')))



    # -----------------------------------------------------------------------------
    # Load dataset with only good stellarators to generate samples
    df = pd.read_csv(FROM_DATA)

    # -----------------------------------------------------------------------------
    # File to keep the new predictions
    fname = TO_DATA
    print('Writing:', fname)

    if os.path.exists(fname):
        f = open(fname, 'a')
    else:
        f = open(fname, 'w')
        print(','.join(df.columns), file=f)
    
    # -----------------------------------------------------------------------------
    # Predict

    it = 0
    progress_bar = tqdm(
        range(NUM_SAMPLES), 
        desc=f"Predicting", 
        leave=False,
        disable=False,
        colour="green"
    )

    for i in progress_bar:
        sample = utils.sample_output(df)

        with torch.no_grad():
            # Transform into tensor
            sample = torch.tensor(sample).float().to(device).unsqueeze(0)

            # Remove mean and divide by std to normalize
            sample = (sample - mean_std["mean"].to(device)) / mean_std["std"].to(device)

            # Pass through model
            sample = model.getMixturesSample(sample, device)

            # Add mean and multiply by std
            sample = sample * mean_std["std_labels"].to(device) + mean_std["mean_labels"].to(device)

            # Run qsc
            sample = sample.cpu().numpy()

            try: 
                sample[0] = utils.round_nfp(sample[0])
                qsc_values = utils.run_qsc(sample[0])

                # Check criteria
                if utils.check_criteria(qsc_values):
                    it += 1

                assert not np.isnan(qsc_values).any()
                assert not np.isinf(qsc_values).any()
                assert not (np.fabs(qsc_values) > np.finfo(np.float32).max).any()

                values = np.concatenate([sample[0], qsc_values], dtype=str)
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
        
    f.close()

            
            
        