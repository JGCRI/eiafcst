"""
Convolutional neural net with auto encoder for predicting GDP from energy use.

Iteration 1
-----------
For our first attempt, we are strictly using the residuals from the
temperature -> electricity model. The goal is to predict quarterly GDP values
from the electricity residuals, using a convolutional neural network with an
auto-encoder.

Caleb Braun
4/25/19
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import time
import os

from eiafcst.dataprep.utils import *
from eiafcst.dataprep.economic import parse_gdp

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from pkg_resources import resource_filename


def prep_data(xpth, ypth=None, train_frac=0.8):
    """
    Combine gas and temperature datasets and prepare them for the model.

    :param xpth:        path to .pkl or .csv file containing electricity
                        residuals by hour
    :param ypth:        path to .csv file containing GDP data, as downloaded
                        from the BEA website
    :param train_frac:  fraction (number between 0-1) of data for training
    """
    HR_WK = 24 * 7  # hours in a week

    xpth = '/Users/brau074/Documents/EIA/eiafcst/eiafcst/models/electricity/elec_model5_residuals.csv'
    elec_residuals = read_training_data(xpth)
    elec_residuals = add_quarter_and_week(elec_residuals, 'time')
    gdp = parse_gdp(syear=elec_residuals['EconYear'].min(), eyear=elec_residuals['EconYear'].max())

    # Display dataset statistics that we'll use for standardization
    gdp_stats = gdp['gdp'].describe()
    print("\nInput dataset summary:")
    print(gdp_stats.transpose(), "\n")

    # Our goal is an array of cases [year-quarters] with dimensions [weeks, hours, regions]
    elec_residuals = elec_residuals.sort_values(['EconYear', 'quarter', 'week', 'time', 'ID'])
    nreg = len(elec_residuals.ID.unique())
    nqtr = len(elec_residuals.EconYear.unique()) * 4

    elec_arr = elec_residuals.residuals.values

    qtr_bounds = np.where(elec_residuals['quarter'].diff())[0]
    qtr_bounds = qtr_bounds[1:]  # Only need inside bounds
    elec_lst = np.array_split(elec_arr, qtr_bounds)
    elec_lst = np.array([a.reshape(-1, HR_WK, nreg) for a in elec_lst])

    assert len(gdp) == nqtr

    # Array with dimensions [weeks, regions]
    return elec_lst, gdp, gdp_stats


def run(trainx, trainy, lr, cl1, cf1, cl2, cf2, l1, l2, epochs, patience, model, plots=True):
    """
    Run the model.

    Does some cool stuff.
    """
    trainx = '/Users/brau074/Documents/EIA/eiafcst/eiafcst/models/electricity/elec_model5_residuals.csv'

    # trainx is reshaped into list of arrays of quarterly vals [weeks, hours, regions]
    trainx, trainy, allstats = prep_data(trainx)

    # Number of aggregate electricity regions
    nhrweek = trainx[0].shape[1]
    nregion = trainx[0].shape[2]
    assert nhrweek == 168

    ncases = len(trainx)

    rand_idx = np.random.permutation(ncases)
    split_idx = int(ncases * 0.75)

    valid_timesteps = rand_idx[split_idx:]
    train_timesteps = rand_idx[:split_idx]
    validx = trainx[valid_timesteps]
    trainx = trainx[train_timesteps]
    validy = trainy.gdp.values[valid_timesteps]
    trainy = trainy.gdp.values[train_timesteps]

    model = build_model(nhrweek, nregion, lr, cl1, cf1, cl2, cf2, l1, l2)

    model.summary()

    plot_model(model, to_file='gdp_model.png', show_shapes=True, show_layer_names=True)

    # The input arrays have 2 dimensions of variable size:
    #  1. The number of cases (quarters)
    #  2. The number of weeks in a case
    # The data therefore cannot be represented as a numpy array because not
    # all dimensions match. To work around this, the data is provided from a
    # generator where each batch is one case, so the input array has a fixed
    # number of weeks (although this number is variable in each batch) and can
    # be represented in a 3d numpy array.
    def batch_generator(inputs, timestep, labels):
        i = 0
        while True:
            batch = inputs[i % len(inputs)]
            # print(f'batch # {i} \t timestep {timestep[i]} \t batch.shape {batch.shape}')
            ts = np.repeat(timestep[i % len(inputs)], batch.shape[0])
            labs = np.repeat(labels[i % len(inputs)], batch.shape[0])
            yield ([batch, ts], [labs, batch])

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(
        monitor='DecoderOutput_loss', patience=patience, restore_best_weights=True)

    model.fit_generator(generator=batch_generator(trainx, train_timesteps, trainy),
                        steps_per_epoch=len(trainx),
                        epochs=epochs,
                        verbose=0,
                        callbacks=[early_stop, PrintDot()],
                        validation_data=batch_generator(validx, valid_timesteps, validy),
                        validation_steps=len(validx))

    return 0, 0, 0, 0


def build_model(nhr, nreg, lr, cl1, cf1, cl2, cf2, l1, l2):
    """
    Build the convolutional neural network.

    Our model is constructed using the Keras functional API
    (https://keras.io/getting-started/functional-api-guide/) which allows us to
    have multiple inputs. Note that all inputs must be specified with an Input
    layer.

    The main input to this model is a 4-dimensional array:
        [cases, weeks, hours, regions]
    where
        cases = number of quarters (open)
        weeks = number of weeks in a quarter (open)
        hours = 168, the number of hours in a week (fixed)
        regions = number of spatial regions (fixed)

    We need an additional inputs:
        timestep = value representing time since the start of the data

    """
    # Weekly timeseries input
    input_numeric = layers.Input(shape=(nhr, nreg), name='HourlyElectricity')

    # The convolutional layers need input tensors with the shape (batch, steps, channels).
    # A convolutional layer is 1D in that it slides through the data length-wise,
    # moving along just one dimension. Our data just has one channel, so we can reshape
    # it to start.
    # input_reshaped = layers.Reshape((nhr, 1))(input_numeric)

    # Conv1D parameters: Conv1D(filters, kernel_size, ...)
    # conv1 = layers.Conv1D(cf1, cl1, padding='same', activation='relu', name='Convolution1')(input_reshaped)
    conv1 = layers.Conv1D(cf1, cl1, padding='same', activation='relu', name='Convolution1')(input_numeric)
    pool1 = layers.MaxPool1D(name='MaxPool1')(conv1)
    conv2 = layers.Conv1D(cf2, cl2, padding='same', activation='relu', name='Convolution2')(pool1)
    pool2 = layers.MaxPool1D(12, name='MaxPool2')(conv2)
    dropo = layers.Dropout(0.5, name='DropHalf')(pool2)
    feature_layer = layers.Flatten()(dropo)

    # Merge the inputs together and end our encoding with fully connected layers
    encoded = layers.Dense(l1, activation='relu')(feature_layer)
    encoded = layers.Dense(l2, activation='relu', name='FinalEncoding')(encoded)

    # At this point, the representation is the most encoded and small
    # Now let's build the decoder
    x = layers.Dense(l1, activation='relu')(encoded)
    x = layers.Dense(cf2 * cl2, activation='relu')(x)
    x = layers.Reshape((cl2, cf2), name='UnFlatten')(x)
    x = layers.Conv1D(filters=cf2, kernel_size=cl2, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(12)(x)
    x = layers.Conv1D(filters=cf1, kernel_size=cl1, padding='same', activation='relu')(x)
    x = layers.UpSampling1D()(x)
    decoded = layers.Conv1D(filters=nreg, kernel_size=cl1, padding='same', activation='relu', name='DecoderOutput')(x)

    # Time since start input (for dealing with energy efficiency changes)
    input_time = layers.Input(shape=(1,), name='TimeSinceStart')

    # This is our actual output, the GDP prediction
    merged_layer = layers.Concatenate()([encoded, input_time])
    output = layers.Dense(8, name='OutputHiddenLayer')(merged_layer)
    output = layers.Dense(1, name='GDP_Output')(output)

    autoencoder = keras.models.Model([input_numeric, input_time], [output, decoded])

    optimizer = tf.keras.optimizers.RMSprop(lr=lr)

    autoencoder.compile(loss='mean_squared_error',
                        optimizer=optimizer,
                        metrics=['mean_absolute_error', 'mean_squared_error'])
    return autoencoder


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('trainx', type=str, help='Training dataset of electricity residuals')
    parser.add_argument('trainy', type=str, help='Training dataset of quarterly GDP values')
    parser.add_argument('-lr', type=float, help='The learning rate (a float)', default=0.01)

    # CNN parameters
    parser.add_argument('-CL1', type=int,
                        help='The number of units in the first convolutional layer\'s kernel (int) [default: 7]',
                        default=7)
    parser.add_argument('-CF1', type=int,
                        help='The number of filters (channels) in the first convolutional layer (int) [default: 3]',
                        default=3)
    parser.add_argument('-CL2', type=int,
                        help='The number of units in the second convolutional layer\'s kernel (int) [default: 7]',
                        default=7)
    parser.add_argument('-CF2', type=int,
                        help='The number of filters (channels) in the second convolutional layer (int) [default: 3]',
                        default=3)

    # Hidden layers parameters
    parser.add_argument('-L1', type=int,
                        help='The number of units in the first hidden layer (int) [default: 16]',
                        default=16)
    parser.add_argument('-L2', type=int,
                        help='The number of units in the first hidden layer (int) [default: 8]',
                        default=8)

    # General model parameters
    parser.add_argument('-epochs', type=int, help='The number of epochs to train for', default=1000)
    parser.add_argument('-patience', type=int,
                        help='How many epochs to continue training without improving dev accuracy (int) [default: 20]',
                        default=20)
    parser.add_argument('-model', type=str,
                        help='Save the best model with this prefix (string) [default: /training_1/model.ckpt]',
                        default=os.path.normpath('/training_1/model.ckpt'))

    return parser.parse_args()


def main():
    """Get hyperparams, load and standardize data, train and evaluate."""
    st = time.time()

    # Hyperparameters passed in via command line
    args = get_args()

    # Set up diagnostics
    hyperparams = [k for k in vars(args).keys()]
    res_metrics = ['mae', 'mse', 'mean train residual', 'mean test residual']
    diag_headers = hyperparams + res_metrics + ['notes']
    diag_fname = diagnostic_file('gdp_results.csv', diag_headers)

    notes = 'Still implementing...'

    # Run model
    results = run(args.trainx, args.trainy, args.lr, args.CL1, args.CF1, args.CL2, args.CF2, args.L1,
                  args.L2, args.epochs, args.patience, args.model, plots=True)

    # Record results
    with open(diag_fname, 'a') as outfile:
        hyper_values = [str(v) for k, v in vars(args).items()]
        diag_results = [str(r) for r in results]
        diag_values = ','.join(hyper_values + diag_results + [notes + '\n'])
        outfile.write(diag_values)

    print(f'Done in {time.time() - st} seconds.')


if __name__ == '__main__':
    main()
