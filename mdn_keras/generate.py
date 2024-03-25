import os
import tqdm

import numpy as np
import pandas as pd

from sampling import *

# -----------------------------------------------------------------------------
# open the output file for writing or appending

fname = 'dataset.csv'

if not os.path.isfile(fname):
    initial = 0
    f = open(fname, 'w')
    fields = ['rc1', 'rc2', 'rc3',
              'zs1', 'zs2', 'zs3',
              'nfp', 'etabar', 'B2c', 'p2',
              'axis_length', 'iota', 'max_elongation',
              'min_L_grad_B', 'min_R0', 'r_singularity',
              'L_grad_grad_B', 'B20_variation', 'beta',
              'DMerc_times_r2']
    print(','.join(fields), file=f)

else:
    df = pd.read_csv(fname)
    initial = df.shape[0]
    f = open(fname, 'a')

# -----------------------------------------------------------------------------
# keep generating until keyboard interrupt

pbar = tqdm.tqdm(total=np.inf, desc='Data counter', initial=initial)

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