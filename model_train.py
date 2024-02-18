import numpy as np
import pandas as pd

from pathlib import Path

from model import create_model, save_weights, callback

# -----------------------------------------------------------------------------

fname = Path('data').joinpath('dataset.csv')
print('Reading:', fname)
df = pd.read_csv(fname)

# -----------------------------------------------------------------------------

nrows = df.shape[0]
nrows = nrows - nrows % 100000
df = df.iloc[:nrows]

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

input_dim = dim
output_dim = dim

model = create_model(input_dim, output_dim)

model.summary()

# -----------------------------------------------------------------------------

batch_size = 1000
epochs = 2000

cb = callback()

try:
    model.fit(X, Y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              validation_split=0.2,
              callbacks=[cb])

except KeyboardInterrupt:
    pass

# -----------------------------------------------------------------------------

model.set_weights(cb.get_weights())

save_weights(model)