import warnings

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

def run_qsc(sample):
    """
    Run Qsc with the given sample and return the output.
    
    Parameters
    ----------
    sample : list
        List of input values for the Qsc (rc1, rc2, rc3, zs1, zs2, zs3, nfp, etabar, B2c, p2).
    
    Returns
    -------
    list
        List of output values from the Qsc (axis_length, iota, max_elongation, min_L_grad_B, min_R0, r_singularity,
        L_grad_grad_B, B20_variation, beta, DMerc_times_r2).
    """

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