import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from qsc_sampling import sample_input, run_qsc

# -----------------------------------------------------------------------------
# set up the output directory, and the output file

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True, parents=True)

fname = DATA_DIR.joinpath('dataset.csv')

# -----------------------------------------------------------------------------
# open the output file for writing or appending

if not fname.exists():
    initial = 0
    f = open(fname, 'w')
    fields_sample = ['rc1', 'rc2', 'rc3',
                     'zs1', 'zs2', 'zs3',
                     'nfp', 'etabar', 'B2c', 'p2']
    fields_output = ['axis_length', 'iota', 'max_elongation',
                     'min_L_grad_B', 'min_R0', 'r_singularity',
                     'L_grad_grad_B', 'B20_variation', 'beta',
                     'DMerc_times_r2']
    fields = fields_sample + fields_output
    print(','.join(fields), file=f)
else:
    initial = pd.read_csv(fname).shape[0]
    f = open(fname, 'a')

# -----------------------------------------------------------------------------
# keep generating until keyboard interrupt

pbar = tqdm(total=np.inf, desc='Data counter', initial=initial)

while True:
    try:
        sample = sample_input()
        output = run_qsc(sample)

        assert not np.isnan(output).any()
        assert not np.isinf(output).any()
        assert not (np.fabs(output) > np.finfo(np.float32).max).any()

        values = np.concatenate([sample, output], dtype=str)
        print(','.join(values), file=f)
        pbar.update(1)

    except Warning:
        continue

    except AssertionError:
        continue

    except KeyboardInterrupt:
        break

pbar.close()

# -----------------------------------------------------------------------------
# close the output file

f.close()

# -----------------------------------------------------------------------------
# try reading the file

df = pd.read_csv(fname)
if len(df) > 0:
    print(df)