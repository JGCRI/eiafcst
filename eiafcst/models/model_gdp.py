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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from pkg_resources import resource_filename


def run(trainx, trainy, lr, cl1, cf1, cl2, cf2, l1, l2, epochs, patience, model, embedsize, plots=True):

    nregion = 14  # Number of aggregate electricity regions
    model = build_model(168, lr, cl1, cf1, cl2, cf2, l1, l2, nregion, embedsize)

    model.summary()

    plot_model(model, to_file='gdp_model.png', show_shapes=True, show_layer_names=True)

    trainx = pd.read_pickle('load_by_agg_region_2006-2017.pkl')
    trainx = read_training_data('load_by_agg_region_2006-2017.csv')

    trainx = trainx[~((trainx['EconYear'] == 2006) & (trainx['quarter'] == 1) & (trainx['week'] == 1))]
    trainx[['NERC Region', 'Hourly Load Data As Of']].groupby('NERC Region').describe()
    return 0, 0, 0, 0


def build_model(input_layer_shape, lr, cl1, cf1, cl2, cf2, l1, l2, embed_in_dim, embed_out_dim):
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

    We need two additional inputs:
        embedding = array of region embeddings
        timestep = value representing time since the start of the data

    """
    # Weekly timeseries input
    input_numeric = layers.Input(shape=(input_layer_shape, ), name='HourlyElectricity')

    # The convolutional layers need input tensors with the shape (batch, steps, channels).
    # A convolutional layer is 1D in that it slides through the data length-wise,
    # moving along just one dimension. Our data just has one channel, so we can reshape
    # it to start.
    input_reshaped = layers.Reshape((input_layer_shape, 1))(input_numeric)

    # Conv1D parameters: Conv1D(filters, kernel_size, ...)
    conv1 = layers.Conv1D(cf1, cl1, padding='same', activation='relu', name='Convolution1')(input_reshaped)
    pool1 = layers.MaxPool1D(name='MaxPool1')(conv1)
    conv2 = layers.Conv1D(cf2, cl2, padding='same', activation='relu', name='Convolution2')(pool1)
    pool2 = layers.MaxPool1D(12, name='MaxPool2')(conv2)
    dropo = layers.Dropout(0.5, name='DropHalf')(pool2)
    feature_layer = layers.Flatten()(dropo)

    # Time since start input (for dealing with energy efficiency changes)
    input_time = layers.Input(shape=(1,), name='TimeSinceStart')

    # Region embbeding input
    input_cat = layers.Input(shape=(1,), name='RegionalEmbedding')
    embed_layer = layers.Embedding(embed_in_dim, embed_out_dim)(input_cat)
    embed_layer = layers.Flatten()(embed_layer)

    # Merge the inputs together and end our encoding with fully connected layers
    merged_layer = layers.Concatenate()([feature_layer, input_time, embed_layer])
    merged_layer = layers.Dense(l1, activation='relu')(merged_layer)
    encoded = layers.Dense(l2, activation='relu')(merged_layer)

    # At this point, the representation is the most encoded and small
    # Now let's build the decoder
    x = layers.Reshape((l2, 1))(encoded)
    x = layers.Conv1D(filters=cf2, kernel_size=cl2, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(12)(x)
    x = layers.Conv1D(filters=cf1, kernel_size=cl1, padding='same', activation='relu')(x)
    x = layers.UpSampling1D()(x)
    x = layers.Conv1D(filters=1, kernel_size=cl1, padding='same', activation='relu')(x)
    decoded = layers.Flatten(name='DecoderOutput')(x)
    # decoder = keras.models.Model(decoder_input, decoded)

    # This is our actual output, the GDP prediction
    output = layers.Dense(1, name='GDP_Output')(encoded)

    autoencoder = keras.models.Model([input_numeric, input_time, input_cat], [output, decoded])

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
                        help='The number of units in the first hidden layer (int) [default: 16]',
                        default=16)

    # General model parameters
    parser.add_argument('-epochs', type=int, help='The number of epochs to train for', default=1000)
    parser.add_argument('-patience', type=int,
                        help='How many epochs to continue training without improving dev accuracy (int) [default: 20]',
                        default=20)
    parser.add_argument('-embedsize', type=int,
                        help='How many dimensions the region embedding should have (int) [default: 5]',
                        default=5)
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
                  args.L2, args.epochs, args.patience, args.model, args.embedsize, plots=True)

    # Record results
    with open(diag_fname, 'a') as outfile:
        hyper_values = [str(v) for k, v in vars(args).items()]
        diag_results = [str(r) for r in results]
        diag_values = ','.join(hyper_values + diag_results + [notes + '\n'])
        outfile.write(diag_values)

    print(f'Done in {time.time() - st} seconds.')


if __name__ == '__main__':
    main()
