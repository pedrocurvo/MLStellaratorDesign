import warnings
import numpy as np
import pandas as pd

from pathlib import Path

from model import create_model

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

fname = Path('data').joinpath('dataset.csv')
print('Reading:', fname)
df = pd.read_csv(fname)

# -----------------------------------------------------------------------------

X_mean = df[df.columns[10:]].mean().values
Y_mean = df[df.columns[:10]].mean().values

X_std = df[df.columns[10:]].std().values
Y_std = df[df.columns[:10]].std().values

# -----------------------------------------------------------------------------

input_dim = 10
output_dim = 10

model = create_model(input_dim, output_dim)

model.summary()

# -----------------------------------------------------------------------------

fname = 'weights.h5'
print('Reading:', fname)
model.load_weights(fname)

# -----------------------------------------------------------------------------

n0 = 0
n1 = 0

while True:
    try:
        axis_length = np.random.uniform(df['axis_length'].min(), df['axis_length'].max())

        iota = np.random.choice([-1., 1.]) * np.random.uniform(0.2, df['iota'].abs().max())

        max_elongation = np.random.uniform(df['max_elongation'].min(), 10.)

        min_L_grad_B = np.random.choice([-1., 1.]) * np.random.uniform(0.1, df['min_L_grad_B'].abs().max())

        min_R0 = np.random.choice([-1., 1.]) * np.random.uniform(0.3, df['min_R0'].abs().max())

        r_singularity = np.random.uniform(0.05, df['r_singularity'].max())

        L_grad_grad_B = np.random.choice([-1., 1.]) * np.random.uniform(0.1, df['L_grad_grad_B'].abs().max())

        B20_variation = np.random.uniform(df['B20_variation'].min(), 5.)

        beta = np.random.uniform(1e-4, df['beta'].max())

        DMerc_times_r2 = np.random.uniform(0., df['DMerc_times_r2'].max())

        assert np.fabs(iota) >= 0.2
        assert max_elongation <= 10.
        assert np.fabs(min_L_grad_B) >= 0.1
        assert np.fabs(min_R0) >= 0.3
        assert r_singularity >= 0.05
        assert np.fabs(L_grad_grad_B) >= 0.1
        assert B20_variation <= 5.
        assert beta >= 1e-4
        assert DMerc_times_r2 >= 0.

        X = np.array([axis_length, iota, max_elongation,
                      min_L_grad_B, min_R0, r_singularity,
                      L_grad_grad_B, B20_variation, beta,
                      DMerc_times_r2])

        X = X - X_mean
        X = X / X_std

        X = np.array([X], dtype=np.float32)
        Y = model.predict(X, batch_size=1, verbose=0)
        Y = np.squeeze(Y)

        Y = Y * Y_std
        Y = Y + Y_mean

        rc1, rc2, rc3 = Y[0], Y[1], Y[2]
        zs1, zs2, zs3 = Y[3], Y[4], Y[5]
        nfp = int(round(Y[6]))
        etabar, B2c, p2 = Y[7], Y[8], Y[9]

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

        passed = True
        try:
            assert np.fabs(iota) >= 0.2
            assert max_elongation <= 10.
            assert np.fabs(min_L_grad_B) >= 0.1
            assert np.fabs(min_R0) >= 0.3
            assert r_singularity >= 0.05
            assert np.fabs(L_grad_grad_B) >= 0.1
            assert B20_variation <= 5.
            assert beta >= 1e-4
            assert DMerc_times_r2 >= 0.
        except AssertionError:
            passed = False

        if passed:
            n1 += 1
        else:
            n0 += 1

        print('\rpassed: %d total: %d percent: %f%%' % (n1, n0+n1, n1/(n0+n1)*100.), end=' ')

    except Warning:
        continue

    except ValueError:
        continue

    except ZeroDivisionError:
        continue

    except KeyboardInterrupt:
        break

print()