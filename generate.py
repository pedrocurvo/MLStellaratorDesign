import numpy as np
import pandas as pd

from qsc import Qsc
from qsc.util import mu0, fourier_minimum

from pathlib import Path

# -----------------------------------------------------------------------------
# set up the output directory
DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True, parents=True)
# set up file name 
fname = DATA_DIR / 'dataset.csv'

# -----------------------------------------------------------------------------
# set up warning behavior, turn warnings into exceptions

import warnings
warnings.filterwarnings('error')

from qsc.newton import logger
def warning(msg, *args, **kwargs):
    raise RuntimeWarning(msg)
logger.warning = warning

# -----------------------------------------------------------------------------

# fname = 'dataset.csv'
# f = open(fname, 'w')

fields = ['rc1', 'rc2', 'rc3', 'zs1', 'zs2', 'zs3', 'nfp', 'etabar', 'B2c', 'p2',
          'iota', 'max_elongation', 'min_L_grad_B', 'min_R0', 'r_singularity',
          'L_grad_grad_B', 'B20_variation', 'beta', 'DMerc_times_r2']

# print(','.join(fields))
# print(','.join(fields), file=f)
if not fname.exists():
    df = pd.DataFrame(columns=fields)
    df.to_csv(fname, index=False)

while True:

    rc1 = np.random.choice([-1., 1.]) * np.random.uniform(0., 1.)
    rc2 = np.random.choice([-1., 1.]) * np.random.uniform(0., np.fabs(rc1))
    rc3 = np.random.choice([-1., 1.]) * np.random.uniform(0., np.fabs(rc2))

    zs1 = np.random.choice([-1., 1.]) * np.random.uniform(0., 1.)
    zs2 = np.random.choice([-1., 1.]) * np.random.uniform(0., np.fabs(zs1))
    zs3 = np.random.choice([-1., 1.]) * np.random.uniform(0., np.fabs(zs2))

    rc = [1., rc1, rc2, rc3]
    zs = [0., zs1, zs2, zs3]

    nfp = np.random.randint(1, 11)

    etabar = np.random.choice([-1., 1.]) * np.random.uniform(0.01, 3.)

    B2c = np.random.choice([-1., 1.]) * np.random.uniform(0.01, 3.)

    p2 = (-1.) * np.random.uniform(0., 4e6)

    order = 'r2'

    try:
        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar, B2c=B2c, p2=p2, order=order)

        iota           = stel.iota
        max_elongation = stel.max_elongation
        min_L_grad_B   = stel.min_L_grad_B
        min_R0         = stel.min_R0
        r_singularity  = stel.r_singularity
        L_grad_grad_B  = fourier_minimum(stel.L_grad_grad_B)
        B20_variation  = stel.B20_variation
        beta           = -mu0 * p2 * stel.r_singularity**2 / stel.B0**2
        DMerc_times_r2 = stel.DMerc_times_r2

        # assert iota >= 0.2
        # assert max_elongation <= 10.
        # assert min_L_grad_B >= 0.1
        # assert min_R0 >= 0.3
        # assert r_singularity >= 0.05
        # assert L_grad_grad_B >= 0.1
        # assert B20_variation <= 5.
        # assert beta >= 1e-4
        # assert DMerc_times_r2 > 0.

        values = [rc1, rc2, rc3, zs1, zs2, zs3, nfp, etabar, B2c, p2,
                  iota, max_elongation, min_L_grad_B, min_R0, r_singularity,
                  L_grad_grad_B, B20_variation, beta, DMerc_times_r2]
        
        df = pd.DataFrame([values], columns=fields)
        df.to_csv(fname, mode='a', header=False, index=False)
        
        # print(','.join([str(value) for value in values]))
        # print(','.join([str(value) for value in values]), file=f)

    except Warning:
        continue

    except AssertionError:
        continue

    except KeyboardInterrupt:
        break

# -----------------------------------------------------------------------------
# try reading the file, to check the results

df = pd.read_csv(fname)
if len(df) > 0:
    print(df)

# -----------------------------------------------------------------------------