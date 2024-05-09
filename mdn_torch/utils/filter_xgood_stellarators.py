import pandas as pd
from qsc import Qsc
from qsc.util import to_Fourier
from tqdm import tqdm

import pandas as pd
import os
from tqdm import tqdm

def filter_xgood_stellarators(input_file: str = None, output_file: str = None):
    """
    Filter the dataset of good stellarators to select only the extra good stellarators.
    The extra good stellarators are the ones that satisfy the following conditions:
        - From the good stellarators:
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
        - Additional conditions:
            - Can be plotted with r=0.1
    
    Parameters
    ----------
    input_file : str
        Path to the input CSV file containing the dataset of good stellarators.
    output_file : str
        Path to the output CSV file where the filtered dataset will be saved.
    """
    # Load dataset with only good stellarators
    df = pd.read_csv(input_file)

    # File to keep the new predictions
    fname = f'{output_file}.csv'
    print('Writing:', fname)

    # Check if file exists
    mode = 'a' if os.path.exists(fname) else 'w'

    # Open the file for writing
    with open(fname, mode) as f:
        if mode == 'w':
            print(','.join(df.columns), file=f)

        for i in tqdm(range(len(df))):
            try:
                sample = df.iloc[i]
                sample = sample.to_numpy()

                # Create Qsc object
                stel = Qsc(rc=[1, *sample[:3]],
                           zs=[0, *sample[3:6]],
                           nfp=sample[6],
                           etabar=sample[7],
                           B2c=sample[8],
                           p2=sample[9],
                           order='r2')

                # Get surface shape at fixed off-axis toroidal angle phi
                R_2D, Z_2D, _ = stel.Frenet_to_cylindrical(r=0.1, ntheta=20)
                # Get Fourier coefficients in order to plot with arbitrary resolution
                RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, stel.nfp, mpol=13, ntor=25, lasym=stel.lasym)

                # Write sample to file
                print(','.join(map(str, sample)), file=f)

            except Exception:
                continue


