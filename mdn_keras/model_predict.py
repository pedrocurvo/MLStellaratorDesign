import os
import tqdm

import numpy as np
import pandas as pd

from model import create_model, load_weights
from sampling import sample_output, round_nfp, run_qsc, check_criteria

# -----------------------------------------------------------------------------

fname = 'dataset.csv'
print('Reading:', fname)
df = pd.read_csv(fname)

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

passed = []

for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    try:
        output = row[dim:].values
        check_criteria(output)
        passed.append(idx)
    except AssertionError:
        continue

df = df.iloc[passed]

# -----------------------------------------------------------------------------

input_dim = dim
output_dim = dim

model = create_model(input_dim, output_dim)

model.summary()

load_weights(model)

# -----------------------------------------------------------------------------

fname = 'predict.csv'
print('Writing:', fname)

if not os.path.isfile(fname):
    f = open(fname, 'w')
    fields = df.columns.values
    print(','.join(fields), file=f)
else:
    f = open(fname, 'a')

# -----------------------------------------------------------------------------

batch_size = 2000

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