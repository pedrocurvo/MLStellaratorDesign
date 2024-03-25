import os
import signal
import multiprocessing

import numpy as np
import pandas as pd

from model import *
from sampling import *

# -----------------------------------------------------------------------------

def ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

cpus = multiprocessing.cpu_count()

# -----------------------------------------------------------------------------

fname = 'dataset.csv'
print('Reading:', fname)
df = pd.read_csv(fname)

# -----------------------------------------------------------------------------

mean = df.mean()
std = df.std()

# -----------------------------------------------------------------------------

dim = 10

X_mean = mean.values[dim:]
Y_mean = mean.values[:dim]

X_std = std.values[dim:]
Y_std = std.values[:dim]

print('X_mean:', X_mean.shape, X_mean.dtype)
print('Y_mean:', Y_mean.shape, Y_mean.dtype)

print('X_std:', X_std.shape, X_std.dtype)
print('Y_std:', Y_std.shape, Y_std.dtype)

# -----------------------------------------------------------------------------

outputs = df[df.columns[dim:]].values.tolist()

with multiprocessing.Pool(cpus, initializer=ignore_sigint) as pool:
    mask = pool.map(check_criteria, outputs)

dataset = df.iloc[mask]

print('dataset:', dataset.shape)

# -----------------------------------------------------------------------------

model = create_model(dim, dim)

load_weights(model)

# -----------------------------------------------------------------------------

fname = 'predict.csv'
print('Writing:', fname)

if not os.path.isfile(fname):
    f = open(fname, 'w')
    print(','.join(df.columns), file=f)
else:
    f = open(fname, 'a')

# -----------------------------------------------------------------------------

with multiprocessing.Pool(cpus, initializer=ignore_sigint) as pool:

    batch_size = 5000

    n_passed = 0
    n_failed = 0

    while n_passed + n_failed < df.shape[0]:
        try:
            X_batch = pool.map(sample_output, [dataset]*batch_size)
            X_batch = np.array(X_batch)
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
                    assert check_criteria(output)
                    n_passed += 1

                except AssertionError:
                    n_failed += 1

                values = np.concatenate([sample, output], dtype=str)
                print(','.join(values), file=f)

                n_total = n_passed + n_failed
                percent = n_passed / n_total * 100.
                print('\rPassed: %d/%d (%.6f %%) ' % (n_passed, n_total, percent), end='')

        except KeyboardInterrupt:
            print()
            break

# -----------------------------------------------------------------------------
# close the output file

f.close()

# -----------------------------------------------------------------------------
# try reading the file

df = pd.read_csv(fname)
if len(df) > 0:
    print(df)