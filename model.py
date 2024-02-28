import time
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback

from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.distributions import Mixture, Categorical, MultivariateNormalTriL
from tensorflow_probability.python.bijectors import FillScaleTriL

# -----------------------------------------------------------------------------

def create_model(input_dim, output_dim):
    model = Sequential()

    # neural network
    model.add(Dense(2048, activation='tanh', input_dim=input_dim))
    model.add(Dense(2048, activation='tanh'))
    
    # number of parameters for each component of the mixture model
    loc_size = output_dim
    scale_size = output_dim * (output_dim + 1) // 2
    params_size = loc_size + scale_size

    # number of components for the mixture model
    K = 10
    units = K + K * params_size
    model.add(Dense(units))
    
    # change for debugging
    validate_args = False
    allow_nan_stats = True

    # mixture model
    model.add(DistributionLambda(lambda t: Mixture(
        # parameterized categorical for component selection
        cat=Categorical(logits=t[...,:K],
                        validate_args=validate_args,
                        allow_nan_stats=allow_nan_stats),
        # parameterized components
        components=[MultivariateNormalTriL(
            # parameterized mean of each component
            loc=t[...,K+i*params_size:K+i*params_size+loc_size],
            # parameterized covariance of each component
            scale_tril=FillScaleTriL().forward(
                -tf.abs(t[...,K+i*params_size+loc_size:K+i*params_size+loc_size+scale_size])),
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats) for i in range(K)],
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats)))

    # optimizer, learning rate, and loss function
    opt = Adam(1e-4)
    loss = lambda y, rv: -rv.log_prob(y)
    model.compile(optimizer=opt, loss=loss)
    
    return model

# -----------------------------------------------------------------------------

def save_weights(model):
    fname = 'model_weights.h5'
    print('Writing:', fname)
    model.save_weights(fname)    

def load_weights(model):
    fname = 'model_weights.h5'
    print('Reading:', fname)
    model.load_weights(fname)

# -----------------------------------------------------------------------------

class callback(Callback):
    
    def on_train_begin(self, logs=None):
        self.min_val_loss = None
        self.min_val_epoch = None
        self.min_val_weights = None
        print('%-10s %10s %10s %10s' % ('time', 'epoch', 'loss', 'val_loss'))
        
    def on_epoch_end(self, epoch, logs=None):
        t = time.strftime('%H:%M:%S')
        loss = logs['loss']
        val_loss = logs['val_loss']
        if (self.min_val_loss == None) or (val_loss < self.min_val_loss):
            self.min_val_loss = val_loss
            self.min_val_epoch = epoch
            self.min_val_weights = self.model.get_weights()
            print('%-10s %10d %10.6f %10.6f *' % (t, epoch, loss, val_loss))
        else:
            print('%-10s %10d %10.6f %10.6f' % (t, epoch, loss, val_loss))
#        if (epoch > 2*self.min_val_epoch):
#            print('Stop training.')
#            self.model.stop_training = True
        if np.isnan(loss) or np.isnan(val_loss):
            self.model.stop_training = True
    def get_weights(self):
        return self.min_val_weights