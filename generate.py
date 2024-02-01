import os
import time
import numpy as np

from qsc import Qsc
from qsc.util import mu0, fourier_minimum

# -----------------------------------------------------------------------------
# set up warning behavior, turn warnings into exceptions

import warnings
warnings.filterwarnings('error')

def warning(msg, *args, **kwargs):
    raise RuntimeWarning(msg)

from qsc.newton import logger as logger1
from qsc.calculate_r2 import logger as logger2
from qsc.calculate_r3 import logger as logger3

logger1.warning = warning
logger2.warning = warning
logger3.warning = warning

# -----------------------------------------------------------------------------
# open the output file for writing or appending

fname = 'dataset.csv'

if not os.path.exists(fname):
    f = open(fname, 'w')

    fields = ['rc1', 'rc2', 'rc3', 'zs1', 'zs2', 'zs3', 'nfp', 'etabar', 'B2c', 'p2',
            'iota', 'max_elongation', 'min_L_grad_B', 'min_R0', 'r_singularity',
            'L_grad_grad_B', 'B20_variation', 'beta', 'DMerc_times_r2']

    print(','.join(fields), file=f)

else:
    f = open(fname, 'a')

# -----------------------------------------------------------------------------
# keep generating until keyboard interrupt

counter = 0
t_start = time.time()

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

#        axis_length    = stel.axis_length
        iota           = stel.iota
        max_elongation = stel.max_elongation
        min_L_grad_B   = stel.min_L_grad_B
        min_R0         = stel.min_R0
        r_singularity  = stel.r_singularity
        L_grad_grad_B  = fourier_minimum(stel.L_grad_grad_B)
        B20_variation  = stel.B20_variation
        beta           = -mu0 * p2 * stel.r_singularity**2 / stel.B0**2
        DMerc_times_r2 = stel.DMerc_times_r2

        # assert np.fabs(iota) >= 0.2
        # assert max_elongation <= 10.
        # assert np.fabs(min_L_grad_B) >= 0.1
        # assert np.fabs(min_R0) >= 0.3
        # assert r_singularity >= 0.05
        # assert np.fabs(L_grad_grad_B) >= 0.1
        # assert B20_variation <= 5.
        # assert beta >= 1e-4
        # assert DMerc_times_r2 > 0.

        values = np.array([rc1, rc2, rc3, zs1, zs2, zs3, nfp, etabar, B2c, p2,
                           iota, max_elongation, min_L_grad_B, min_R0, r_singularity,
                           L_grad_grad_B, B20_variation, beta, DMerc_times_r2])

        assert not np.isnan(values).any()
        assert not np.isinf(values).any()
        assert not (np.fabs(values) > np.finfo(np.float32).max).any()

        counter += 1
        seconds = time.time() - t_start
        print('\r%d samples %d secs %.1f samples/sec ' % (counter, seconds, counter/seconds), end=' ')

        print(','.join(values.astype(str)), file=f)

    except Warning:
        continue

    except AssertionError:
        continue

    except KeyboardInterrupt:
        break

print()

# -----------------------------------------------------------------------------
# close the output file

f.close()
