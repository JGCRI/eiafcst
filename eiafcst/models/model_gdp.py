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


def build_model(input_layer_shape, l1, l2, lr, embed_in_dim, embed_out_dim):
    """
    Build the convolutional neural network.

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
    input_num = layers.Input(shape=(input_layer_shape, 1))
    conv1 = layers.Conv1D(filters=5, kernel_size=7, padding='same', activation='relu')(input_num)
    pool1 = layers.MaxPool1D()(conv1)
    conv2 = layers.Conv1D(filters=1, kernel_size=4, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPool1D(12)(conv2)
    dropo = layers.Dropout(0.5)(pool2)
    feature_layer = layers.Flatten()(dropo)

    # Time since start input (for dealing with energy efficiency changes)
    input_time = layers.Input(shape=(1,))

    # Region embbeding input
    input_cat = layers.Input(shape=(1,))
    embed_layer = layers.Embedding(embed_in_dim, embed_out_dim)(input_cat)
    embed_layer = layers.Flatten()(embed_layer)

    merged_layer = layers.Concatenate()([feature_layer, input_time, embed_layer])
    output = layers.Dense(l1, activation='relu')(merged_layer)
    output = layers.Dense(l2, activation='relu')(output)
    output = layers.Dense(1, bias_initializer=tf.keras.initializers.constant(4.0))(output)

    model = keras.models.Model(inputs=[input_num, input_time, input_cat], outputs=[output])

    optimizer = tf.keras.optimizers.RMSprop(lr=lr)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('trainx', type=str, help='Training dataset of gas consumption values')
    parser.add_argument('trainy', type=str, help='Training dataset of temperature values')
    parser.add_argument('-lr', type=float, help='The learning rate (a float)', default=0.01)
    parser.add_argument('-L1', type=int,
                        help='The number of units in the first hidden layer (int) [default: 16]',
                        default=16)
    parser.add_argument('-L2', type=int,
                        help='The number of units in the first hidden layer (int) [default: 16]',
                        default=16)
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
    results = run(args.trainx, args.trainy, args.lr, args.L1, args.L2, args.epochs,
                  args.patience, args.model, args.embedsize, plots=True)

    # Record results
    with open(diag_fname, 'a') as outfile:
        hyper_values = [str(v) for k, v in vars(args)]
        diag_results = [str(r) for r in results]
        diag_values = ','.join(hyper_values + diag_results + [notes + '\n'])
        outfile.write(diag_values)

    print(f'Done in {time.time() - st} seconds.')


if __name__ == '__main__':
    main()
