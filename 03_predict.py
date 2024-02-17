import numpy as np
import pandas as pd

from model import create_model

# -----------------------------------------------------------------------------

fname = 'dataset.csv'
print('Reading:', fname)
df = pd.read_csv(fname)

# -----------------------------------------------------------------------------

mean = df.mean()
std = df.std()

# -----------------------------------------------------------------------------

input_dim = 10
output_dim = 10

model = create_model(input_dim, output_dim)

# -----------------------------------------------------------------------------

fname = 'weights.h5'
print('Reading:', fname)
model.load_weights(fname)

# -----------------------------------------------------------------------------
