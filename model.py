import time

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import Callback

from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.distributions import MultivariateNormalTriL
from tensorflow_probability.python.bijectors import FillScaleTriL, Exp

# -----------------------------------------------------------------------------

def create_model(input_dim, output_dim):
    model = Sequential()

    model.add(Dense(256, activation='tanh', input_dim=input_dim))
    model.add(Dense(256, activation='tanh'))

    units = output_dim + output_dim * (output_dim + 1) // 2
    model.add(Dense(units))

    model.add(DistributionLambda(
        make_distribution_fn=lambda t: MultivariateNormalTriL(
            loc=t[...,:output_dim],
            scale_tril=FillScaleTriL(diag_bijector=Exp(),
                                     diag_shift=None).forward(t[...,output_dim:]))))

    model.summary()
    
    fname = 'model.png'
    print('Writing:', fname)
    plot_model(model, to_file=fname, show_shapes=True, show_layer_names=False)

    opt = Adam(1e-4)
    loss = lambda y, rv: -rv.log_prob(y)
    model.compile(optimizer=opt, loss=loss)
    
    return model

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
        if epoch > 2*self.min_val_epoch:
            print('Stop training.')
            self.model.stop_training = True

    def get_weights(self):
        return self.min_val_weights