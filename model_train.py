import numpy as np
import pandas as pd

from pathlib import Path

from model import create_model, save_weights, callback

# -----------------------------------------------------------------------------

fname = Path('data').joinpath('dataset_2M.csv')
print('Reading:', fname)
df = pd.read_csv(fname)

# -----------------------------------------------------------------------------

mean = df.mean()
std = df.std()

df = df - mean
df = df / std

# -----------------------------------------------------------------------------

dim = 10

X = df[df.columns[dim:]].values.astype(np.float32)
Y = df[df.columns[:dim]].values.astype(np.float32)

print('X:', X.shape, X.dtype)
print('Y:', Y.shape, Y.dtype)

# -----------------------------------------------------------------------------

N = 5

r = np.arange(X.shape[0])

X_train = X[r % N != 0]
Y_train = Y[r % N != 0]

X_valid = X[r % N == 0]
Y_valid = Y[r % N == 0]

print('X_train:', X_train.shape, X_train.dtype)
print('Y_train:', Y_train.shape, Y_train.dtype)

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

# -----------------------------------------------------------------------------

input_dim = dim
output_dim = dim

model = create_model(input_dim, output_dim)

model.summary()

# -----------------------------------------------------------------------------

batch_size = 2000
print('batch_size:', batch_size)

n_train = (X_train.shape[0] // batch_size) * batch_size
n_valid = (X_valid.shape[0] // batch_size) * batch_size

X_train = X_train[:n_train]
Y_train = Y_train[:n_train]

X_valid = X_valid[:n_valid]
Y_valid = Y_valid[:n_valid]

print('X_train:', X_train.shape, X_train.dtype)
print('Y_train:', Y_train.shape, Y_train.dtype)

print('X_valid:', X_valid.shape, X_valid.dtype)
print('Y_valid:', Y_valid.shape, Y_valid.dtype)

# -----------------------------------------------------------------------------

epochs = 2000

cb = callback()

try:
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_data=(X_valid, Y_valid),
              callbacks=[cb])

except KeyboardInterrupt:
    pass

# -----------------------------------------------------------------------------

model.set_weights(cb.get_weights())

save_weights(model)