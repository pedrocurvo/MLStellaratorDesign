import os
import sys
# Get the parent directory of the current directory (external_package)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_dir)
print(parent_dir)

import torch
import torch_optimizer as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from StellaratorDataSet import StellaratorDataSetInverse
from timeit import default_timer as timer
from datetime import datetime
from train_pipeline import engine, utils, data_setup
from MDNFullCovariance import MDNFullCovariance
from preditcions_utils import sample_output, check_criteria, run_qsc, round_nfp
from multiprocessing import Pool

def process_iteration(_):
    sample = sample_output(df)

    with torch.no_grad():
        sample = torch.tensor(sample).float().to(device).unsqueeze(0)
        sample = (sample - mean_std["mean"].to(device)) / mean_std["std"].to(device)
        output = model.getMixturesSample(sample, device)
        output = output * mean_std["std_labels"].to(device) + mean_std["mean_labels"].to(device)
        output[0][6] = torch.round(output[0][6])
        output = output.cpu().numpy()

        try: 
            output[0] = round_nfp(output[0])
            qsc_values = run_qsc(output[0])

            if check_criteria(qsc_values):
                return np.concatenate([output[0], qsc_values], dtype=str)

        except (Warning, AssertionError):
            pass

    return None

if __name__ == "__main__":
    full_dataset = StellaratorDataSetInverse(npy_file='../data/dataset.npy')
    
    if torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    BATCH_SIZE = 500
    NUM_OF_WORKERS = os.cpu_count()
    
    train_dataloader, val_dataloader, test_dataloader, mean_std = data_setup.create_dataloaders(dataset=full_dataset,
                                                                                                train_size=0.5,
                                                                                                val_size=0.2,
                                                                                                batch_size=BATCH_SIZE,
                                                                                                num_workers=NUM_OF_WORKERS)

    model = MDNFullCovariance(input_dim=10,
                               output_dim=10,
                               num_gaussians=62).to(device)
    model.load_state_dict(torch.load("models/MDNFullCovariance/2024_03_28_11_53_42.pth"))

    df = pd.read_csv('../data/good_stellarators_dataset_first.csv')
    fname = '../data/predict_first.csv'
    print('Writing:', fname)

    with open(fname, 'w') as f:
        print(','.join(df.columns), file=f)

        # Create a Pool of workers
        with Pool(processes=os.cpu_count()) as pool:
            results = list(tqdm(pool.imap(process_iteration, range(6000000)), total=6000000))

        for values in results:
            if values is not None:
                print(','.join(values), file=f)
