import numpy as np

from timeit import default_timer as timer

from qsc import Qsc
from qsc.util import mu0, fourier_minimum

import warnings
warnings.filterwarnings('error')

# -----------------------------------------------------------------------------

rc_min = [1.0, 0.01]
rc_max = [1.0, 0.20]

zs_min = [0.0, -0.20]
zs_max = [0.0, -0.01]

nfp_min = 2
nfp_max = 2

etabar_min = 0.01
etabar_max = 3.00

B2c_min = 0.
B2c_max = 0.

p2_min = 0.
p2_max = 0.

keep_all = False
max_seconds = 10

# -----------------------------------------------------------------------------

min_R0_to_keep            = 0.3
max_elongation_to_keep    = 10
min_iota_to_keep          = 0.2
min_L_grad_B_to_keep      = 0.1
min_L_grad_grad_B_to_keep = 0.1
max_B20_variation_to_keep = 5.0
min_r_singularity_to_keep = 0.05
min_beta_to_keep          = 1e-4

# -----------------------------------------------------------------------------

print('%10s %10s %10s %10s %10s %10s %10s %10s' % ('rc1',
                                                   'zs1',
                                                   'nfp',
                                                   'etabar',
                                                   'iota',
                                                   'min_R0',
                                                   'max_elong',
                                                   'L_grad_B'))

t0 = timer()

while (timer() - t0) < max_seconds:
    rc = [np.random.uniform(a, b) for a, b in zip(rc_min, rc_max)]
    zs = [np.random.uniform(a, b) for a, b in zip(zs_min, zs_max)]
    nfp = np.random.randint(nfp_min, nfp_max + 1)
    etabar = np.random.uniform(etabar_min, etabar_max)
    order = np.random.choice(['r1', 'r2'])
    B2c = np.random.uniform(B2c_min, B2c_max)
    p2 = np.random.uniform(p2_min, p2_max)
    try:
        q = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar, order=order, B2c=B2c, p2=p2)
        if not keep_all:
            if order == 'r1':
                if q.min_R0 < min_R0_to_keep:
                    continue
                if q.max_elongation > max_elongation_to_keep:
                    continue
                if q.iota < min_iota_to_keep:
                    continue
            if order == 'r2':
                if q.min_L_grad_B < min_L_grad_B_to_keep:
                    continue
                if fourier_minimum(q.L_grad_grad_B) < min_L_grad_grad_B_to_keep:
                    continue
                if q.B20_variation > max_B20_variation_to_keep:
                    continue
                beta = -mu0 * p2 * q.r_singularity**2 / q.B0**2
                if beta < min_beta_to_keep:
                    continue
        print('%10.3f %10.3f %10d %10.3f %10.2f %10.1f %10.1f %10.2f' % (rc[1],
                                                                         zs[1],
                                                                         nfp,
                                                                         etabar,
                                                                         q.iota,
                                                                         q.min_R0,
                                                                         q.max_elongation,
                                                                         q.min_L_grad_B))
    except Warning:
        continue