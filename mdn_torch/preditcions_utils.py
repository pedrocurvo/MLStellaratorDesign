import warnings
import numpy as np

from qsc import Qsc
from qsc.util import mu0, fourier_minimum
from qsc.newton import logger as logger1
from qsc.calculate_r2 import logger as logger2
from qsc.calculate_r3 import logger as logger3

# -----------------------------------------------------------------------------
# set up warning behavior, turn warnings into exceptions

warnings.filterwarnings('error')

def warning(msg, *args, **kwargs):
    raise RuntimeWarning(msg)

logger1.warning = warning
logger2.warning = warning
logger3.warning = warning

# -----------------------------------------------------------------------------

def round_nfp(sample):
    nfp = sample[6]
    nfp = np.clip(nfp, 1., None)
    nfp = np.round(nfp)
    nfp = nfp.astype(int)
    sample[6] = nfp
    return sample


# -----------------------------------------------------------------------------

def run_qsc(sample):

    [rc1, rc2, rc3, zs1, zs2, zs3, nfp, etabar, B2c, p2] = sample

    rc = [1., rc1, rc2, rc3]
    zs = [0., zs1, zs2, zs3]

    order = 'r2'

    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar, B2c=B2c, p2=p2, order=order)

    axis_length    = stel.axis_length
    iota           = stel.iota
    max_elongation = stel.max_elongation
    min_L_grad_B   = stel.min_L_grad_B
    min_R0         = stel.min_R0
    r_singularity  = stel.r_singularity
    L_grad_grad_B  = fourier_minimum(stel.L_grad_grad_B)
    B20_variation  = stel.B20_variation
    beta           = -mu0 * p2 * stel.r_singularity**2 / stel.B0**2
    DMerc_times_r2 = stel.DMerc_times_r2

    output = [axis_length, iota, max_elongation,
              min_L_grad_B, min_R0, r_singularity,
              L_grad_grad_B, B20_variation, beta,
              DMerc_times_r2]

    return output

# -----------------------------------------------------------------------------

def check_criteria(output):

    [axis_length, iota, max_elongation,
     min_L_grad_B, min_R0, r_singularity,
     L_grad_grad_B, B20_variation, beta,
     DMerc_times_r2] = output

    try:
        assert axis_length > 0.
        assert np.fabs(iota) >= 0.2
        assert max_elongation <= 10.
        assert np.fabs(min_L_grad_B) >= 0.1
        assert np.fabs(min_R0) >= 0.3
        assert r_singularity >= 0.05
        assert np.fabs(L_grad_grad_B) >= 0.1
        assert B20_variation <= 5.
        assert beta >= 1e-4
        assert DMerc_times_r2 > 0.
        return True

    except AssertionError:
        return False

# -----------------------------------------------------------------------------

def sample_output(dataset):
    cols = ['axis_length', 'iota', 'max_elongation',
            'min_L_grad_B', 'min_R0', 'r_singularity',
            'L_grad_grad_B', 'B20_variation', 'beta',
            'DMerc_times_r2']

    n = dataset.shape[0]

    row_index = np.random.randint(n)  # Randomly select a row index

    sample = [dataset[col].iloc[np.random.randint(n)] if col != 'r_singularity' and col != 'beta'
              else dataset[col].iloc[row_index] for col in cols]

    return sample