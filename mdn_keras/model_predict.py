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

size = df.shape[0]
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

def get_criteria_mask(df):
    outputs = df[df.columns[dim:]].values.tolist()
    with multiprocessing.Pool(cpus, initializer=ignore_sigint) as pool:
        mask = pool.map(check_criteria, outputs)
    return mask

mask = get_criteria_mask(df)

dataset = df.iloc[mask]

print('dataset:', dataset.shape)

# -----------------------------------------------------------------------------

model = create_model(dim, dim)

load_weights(model)

# -----------------------------------------------------------------------------

fname = 'predict.csv'
print('Writing:', fname)

if not os.path.isfile(fname):
    n_passed = 0
    n_failed = 0
    f = open(fname, 'w')
    print(','.join(df.columns), file=f)
else:
    df = pd.read_csv(fname)
    mask = get_criteria_mask(df)
    n_passed = mask.count(True)
    n_failed = mask.count(False)
    f = open(fname, 'a')

# -----------------------------------------------------------------------------

batch_size = 5000

with multiprocessing.Pool(cpus, initializer=ignore_sigint) as pool:
    while n_passed + n_failed < size:
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

                    assert not np.isnan(output).any()
                    assert not np.isinf(output).any()
                    assert not (np.fabs(output) > np.finfo(np.float32).max).any()

                except Warning:
                    continue

                except AssertionError:
                    continue

                values = np.concatenate([sample, output], dtype=str)
                print(','.join(values), file=f)

                if check_criteria(output):
                    n_passed += 1
                else:
                    n_failed += 1

                n_total = n_passed + n_failed
                percent = n_passed / n_total * 100.
                print('\rPassed: %d/%d (%.6f %%) ' % (n_passed, n_total, percent), end='')

        except KeyboardInterrupt:
            break

print()

# -----------------------------------------------------------------------------

f.close()

# -----------------------------------------------------------------------------

df = pd.read_csv(fname)
if len(df) > 0:
    print(df)