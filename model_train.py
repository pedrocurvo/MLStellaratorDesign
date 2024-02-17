import numpy as np
import pandas as pd

from pathlib import Path

from model import create_model, callback

# -----------------------------------------------------------------------------

fname = Path('data').joinpath('dataset.csv')
print('Reading:', fname)
df = pd.read_csv(fname)

# -----------------------------------------------------------------------------

mean = df.mean()
std = df.std()

df = df - mean
df = df / std

# -----------------------------------------------------------------------------

X = df[df.columns[10:]].values.astype(np.float32)
Y = df[df.columns[:10]].values.astype(np.float32)

print('X:', X.shape, X.dtype)
print('Y:', Y.shape, Y.dtype)

# -----------------------------------------------------------------------------

input_dim = X.shape[1]
output_dim = Y.shape[1]

model = create_model(input_dim, output_dim)

model.summary()

# -----------------------------------------------------------------------------

batch_size = 200
epochs = 1000

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

model.set_weights(cb.get_weights())

# -----------------------------------------------------------------------------

fname = 'weights.h5'
print('Writing:', fname)
model.save_weights(fname)