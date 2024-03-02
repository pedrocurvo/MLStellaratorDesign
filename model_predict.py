import numpy as np
import pandas as pd

from pathlib import Path

from model import create_model, load_weights
from sampling import sample_output, round_nfp, run_qsc, check_criteria

# -----------------------------------------------------------------------------

fname = Path('data').joinpath('dataset_2M.csv')
print('Reading:', fname)
df = pd.read_csv(fname)

# -----------------------------------------------------------------------------

nrows = df.shape[0]
nrows = nrows - nrows % 10000
df = df.iloc[:nrows]

# -----------------------------------------------------------------------------

mean = df.mean()
std = df.std()

# -----------------------------------------------------------------------------

dim = 10

X_mean = mean[dim:].values
Y_mean = mean[:dim].values

X_std = std[dim:].values
Y_std = std[:dim].values

# -----------------------------------------------------------------------------

input_dim = dim
output_dim = dim

model = create_model(input_dim, output_dim)

load_weights(model)

model.summary()

# -----------------------------------------------------------------------------

fname = Path('data').joinpath('predict.csv')
print('Writing:', fname)

if not fname.exists():
    f = open(fname, 'w')
    fields_input = ['rc1', 'rc2', 'rc3',
                    'zs1', 'zs2', 'zs3',
                    'nfp', 'etabar', 'B2c', 'p2']
    fields_output = ['axis_length', 'iota', 'max_elongation',
                     'min_L_grad_B', 'min_R0', 'r_singularity',
                     'L_grad_grad_B', 'B20_variation', 'beta',
                     'DMerc_times_r2']
    fields = fields_input + fields_output
    print(','.join(fields), file=f)
else:
    f = open(fname, 'a')

# -----------------------------------------------------------------------------

batch_size = 200

n_passed = 0
n_failed = 0

while True:
    try:
        X_batch = np.array([sample_output(df) for _ in range(batch_size)])
        X_batch = X_batch - X_mean
        X_batch = X_batch / X_std
        X_batch = X_batch.astype(np.float32)

        Y_batch = model.predict(X_batch, batch_size=batch_size, verbose=0)
        Y_batch = Y_batch * Y_std
        Y_batch = Y_batch + Y_mean

        for sample in Y_batch:
            try:
                sample = round_nfp(sample)
                output = run_qsc(sample)
            except Warning:
                continue

            try:
                assert not np.isnan(output).any()
                assert not np.isinf(output).any()

                check_criteria(output)
                n_passed += 1

                values = np.concatenate([sample, output], dtype=str)
                print(','.join(values), file=f)

            except AssertionError:
                n_failed += 1

            n_total = n_passed + n_failed
            percent = n_passed / n_total * 100.
            print('\rPassed: %d/%d (%.6f %%) ' % (n_passed, n_total, percent), end='')

    except KeyboardInterrupt:
        break

print()

# -----------------------------------------------------------------------------
# close the output file

f.close()

# -----------------------------------------------------------------------------
# try reading the file

df = pd.read_csv(fname)
if len(df) > 0:
    print(df)