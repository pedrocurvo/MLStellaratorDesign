import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from MDNFullCovariance import MDNFullCovariance
from utils import check_criteria, run_qsc, sample_output, round_nfp


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"Using device: {device}")
    
    # Load mean_std from a file
    mean_std = torch.load("models/mean_std_2024_04_05_01_01_18.pth", map_location=torch.device('cpu'))

    # Create model
    model = MDNFullCovariance(input_dim=10,
                            output_dim=10,
                            num_gaussians=62
    ).to(device)
    
    # Load a previous model (optional: uncomment if you want to load a previous model): transfer learning
    # model.load_state_dict(torch.load("models/MDNFullCovariance/2024_03_28_11_53_42.pth")) # 1
    # model.load_state_dict(torch.load("models/MDNFullCovariance/2024_03_30_02_40_44.pth")) # 2
    # model.load_state_dict(torch.load("models/MDNFullCovariance/2024_04_02_15_47_52.pth")) # 3
    # model.load_state_dict(torch.load("models/MDNFullCovariance/2024_04_03_10_03_21.pth")) # 4
    # model.load_state_dict(torch.load("models/MDNFullCovariance/2024_04_04_01_37_38.pth", map_location=torch.device('cpu'))) # 5
    # model.load_state_dict(torch.load("models/MDNFullCovariance/2024_04_04_13_41_19.pth")) # combined
    model.load_state_dict(torch.load("models/MDNFullCovariance/2024_04_05_01_01_18.pth", map_location=torch.device('cpu'))) # 5



    # -----------------------------------------------------------------------------
    # Load dataset with only good stellarators to generate samples
    df = pd.read_csv('../data/6_dataset/sixth_good_stels.csv')

    # -----------------------------------------------------------------------------
    # File to keep the new predictions
    fname = './dump/seventh_dataset.csv'
    print('Writing:', fname)

    current_file_len = 0
    if os.path.exists(fname):
        f = open(fname, 'a')
        current_file_len = sum(1 for line in open(fname))
    else:
        f = open(fname, 'w')
        print(','.join(df.columns), file=f)
    
    # -----------------------------------------------------------------------------
    # Predict

    it = 0
    progress_bar = tqdm(
        range(100000 - current_file_len), 
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

            # Remove mean and divide by std to normalize
            sample = (sample - mean_std["mean"].to(device)) / mean_std["std"].to(device)

            # Pass through model
            sample = model.getMixturesSample(sample, device)

            # Add mean and multiply by std
            sample = sample * mean_std["std_labels"].to(device) + mean_std["mean_labels"].to(device)

            # Run qsc
            sample = sample.cpu().numpy()

            try: 
                sample[0] = round_nfp(sample[0])
                qsc_values = run_qsc(sample[0])

                # Check criteria
                if check_criteria(qsc_values):
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

            
            
        