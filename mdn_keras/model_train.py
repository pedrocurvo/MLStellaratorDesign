import numpy as np
import pandas as pd

from keras.callbacks import TensorBoard

from model import create_model, load_weights, save_weights, callback

# -----------------------------------------------------------------------------

fname = 'dataset.csv'
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

steps_per_epoch = 500

batch_size = X_train.shape[0] // steps_per_epoch
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

initial_rate = 1e-4

learning_rate = initial_rate

interrupt = False

while (learning_rate >= 1e-7) and (interrupt == False):
    print('learning_rate:', learning_rate)
    model = create_model(X.shape[1], Y.shape[1], learning_rate)
    model.summary()

    load_weights(model)

    epochs = 200
    cb = callback()
    tb = TensorBoard(write_graph=False)

    try:
        model.fit(X_train, Y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=0,
                 validation_data=(X_valid, Y_valid),
                 callbacks=[cb, tb])
    except KeyboardInterrupt:
        interrupt = True

    save_weights(model)

    learning_rate = np.round(learning_rate / 10.**0.25, 15)