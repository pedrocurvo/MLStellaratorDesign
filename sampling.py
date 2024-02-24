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

def sample_input():

    rc1 = np.random.choice([-1., 1.]) * np.random.uniform(0., 1.)
    rc2 = np.random.choice([-1., 1.]) * np.random.uniform(0., np.fabs(rc1))
    rc3 = np.random.choice([-1., 1.]) * np.random.uniform(0., np.fabs(rc2))

    zs1 = np.random.choice([-1., 1.]) * np.random.uniform(0., 1.)
    zs2 = np.random.choice([-1., 1.]) * np.random.uniform(0., np.fabs(zs1))
    zs3 = np.random.choice([-1., 1.]) * np.random.uniform(0., np.fabs(zs2))

    nfp = np.random.randint(1, 11)

    etabar = np.random.choice([-1., 1.]) * np.random.uniform(0.01, 3.)

    B2c = np.random.choice([-1., 1.]) * np.random.uniform(0.01, 3.)

    p2 = (-1.) * np.random.uniform(0., 4e6)

    sample = [rc1, rc2, rc3, zs1, zs2, zs3, nfp, etabar, B2c, p2]

    return sample

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

# -----------------------------------------------------------------------------

def sample_output(dataset):

    axis_length = dataset['axis_length'].values
    axis_length = axis_length[axis_length > 0.]
    axis_length = np.random.choice(axis_length)

    iota = dataset['iota'].values
    iota = iota[np.fabs(iota) >= 0.2]
    iota = np.random.choice(iota)

    max_elongation = dataset['max_elongation'].values
    max_elongation = max_elongation[max_elongation <= 10.]
    max_elongation = np.random.choice(max_elongation)

    min_L_grad_B = dataset['min_L_grad_B'].values
    min_L_grad_B = min_L_grad_B[np.fabs(min_L_grad_B) >= 0.1]
    min_L_grad_B = np.random.choice(min_L_grad_B)

    min_R0 = dataset['min_R0'].values
    min_R0 = min_R0[np.fabs(min_R0) >= 0.3]
    min_R0 = np.random.choice(min_R0)

    r_singularity = dataset['r_singularity'].values
    r_singularity = r_singularity[r_singularity >= 0.05]
    r_singularity = np.random.choice(r_singularity)

    L_grad_grad_B = dataset['L_grad_grad_B'].values
    L_grad_grad_B = L_grad_grad_B[np.fabs(L_grad_grad_B) >= 0.1]
    L_grad_grad_B = np.random.choice(L_grad_grad_B)
    
    B20_variation = dataset['B20_variation'].values
    B20_variation = B20_variation[B20_variation <= 5.]
    B20_variation = np.random.choice(B20_variation)

    beta = dataset['beta'].values
    beta = beta[beta >= 1e-4]
    beta = np.random.choice(beta)

    DMerc_times_r2 = dataset['DMerc_times_r2'].values
    DMerc_times_r2 = DMerc_times_r2[DMerc_times_r2 > 0.]
    DMerc_times_r2 = np.random.choice(DMerc_times_r2)

    sample = [axis_length, iota, max_elongation,
              min_L_grad_B, min_R0, r_singularity,
              L_grad_grad_B, B20_variation, beta,
              DMerc_times_r2]

    return sample