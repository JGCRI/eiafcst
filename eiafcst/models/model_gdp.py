"""
Convolutional neural net with auto encoder for predicting GDP from energy use.

Iteration 2
-----------
For our second attempt, we are using the residuals from the temperature
and electricity model, as well as natural gas and petroleum inputs. The goal
is to predict quarterly GDP values from these inputs, using a convolutional
neural network with an auto-encoder.

Caleb Braun
4/25/19
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os

from pkg_resources import resource_filename

from eiafcst.dataprep.utils import plot_history, DiagnosticFile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def unstandardize(gdp_orig, val):
    """Convert standardized GDP to dollar values."""
    return round(val * gdp_orig.std() + gdp_orig.mean(), 2)


def load_inputs(dsets):
    """
    Load model inputs.

    :param dsets:   List of names of datasets to load ('train', 'dev', or 'test')
    """
    assert isinstance(dsets, list)

    inputs = []

    for dset in dsets:
        input_dir = resource_filename('eiafcst', os.path.join('models', 'gdp', dset))
        dset_files = os.listdir(input_dir)
        inputs.append({f.replace('.npy', ''): np.load(os.path.join(input_dir, f)) for f in dset_files})

    return inputs


def train_model(train, dev, hpars, model, plots=True):
    """
    Run the deep learning model.

    Trains and evaluates the model. Assumes all inputs have been verified at
    this point.

    :param train:       Dictionary of all training datasets for all inputs and outputs.
    :param dev:         Dictionary of all validation datasets for all inputs and outputs.
    :param hpars:       Dictionary of all hyperparameters.
    :param model:       Location to store best model; if empty, does not save.
    :param plots:       Boolean; generate plots of model performance? [default: True]
    """
    # standardize GDP values (elec values already standardized from prev. model)
    train_labels = (train['gdp'] - train['gdp'].mean()) / train['gdp'].std()
    dev_labels = (dev['gdp'] - dev['gdp'].mean()) / dev['gdp'].std()

    # Number of hours (168), aggregate electricity regions (14), and cases
    nhrweek = train['elec'][0].shape[1]
    nregion = train['elec'][0].shape[2]
    assert nhrweek == 168

    model = build_model(nhrweek, nregion, hpars.lr, hpars.wgdp, hpars.wdec, hpars.C, hpars.L1, hpars.L2, hpars.lgdp)

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
    def batch_generator(ele_residuals, timestep, gas_residuals, pet_residuals, labels):
        i = 0
        while True:
            # print(f'Input length: {len(elec_residuals)}\tBATCH #{i % len(elec_residuals)}')
            ele = ele_residuals[i % len(ele_residuals)]
            gas = gas_residuals[i % len(ele_residuals)]
            pet = pet_residuals[i % len(ele_residuals)]
            ts = np.repeat(timestep[i % len(ele_residuals)], ele.shape[0])
            labs = np.repeat(labels[i % len(ele_residuals)], ele.shape[0])
            yield ([ele, ts, gas, pet], [labs, ele])
            i += 1

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=hpars.patience, restore_best_weights=True)

    history = model.fit_generator(generator=batch_generator(train['elec'], train['time'], train['gas'],
                                                            train['petrol'], train_labels),
                                  steps_per_epoch=len(train['elec']),
                                  epochs=hpars.epochs,
                                  verbose=0,
                                  callbacks=[early_stop],
                                  validation_data=batch_generator(dev['elec'], dev['time'], dev['gas'],
                                                                  dev['petrol'], dev_labels),
                                  validation_steps=len(dev['elec']))

    plot_history(history,
                 cols=['DecoderOut_mean_absolute_error', 'val_DecoderOut_mean_absolute_error',
                       'GDP_Output_mean_absolute_error', 'val_GDP_Output_mean_absolute_error'],
                 labs=['Train Decoder Error', 'Val Decoder Error', 'Train GDP Error', 'Val GDP Error'],
                 savefile='gdp_train_history.png')

    # Evaluate the model on our validation set
    metrics = model.evaluate_generator(generator=batch_generator(dev['elec'], dev['time'], dev['gas'],
                                                                 dev['petrol'], dev_labels),
                                       steps=len(dev['elec']))

    gdp_mae = metrics[3]
    dec_mae = metrics[5]

    train_predictions, train_residuals = run_prediction(model, train, 'training', train_labels)
    dev_predictions, dev_residuals = run_prediction(model, dev, 'development', dev_labels)

    dev_resid_abs_mean = np.abs(dev_residuals).mean()
    train_resid_abs_mean = np.abs(train_residuals).mean()
    print(f"Validation set residuals absolute mean {dev_resid_abs_mean}")
    print(f"Training set residuals absolute mean {train_resid_abs_mean}")

    # Plot residuals
    plt.xlabel('Timestep')
    plt.ylabel('Prediction residual (Billion USD)')
    plt.scatter(dev['time'], dev_residuals, c='#ef8a62', label='development')
    plt.scatter(train['time'], train_residuals, c='#67a9cf', label='train')
    plt.title('Residuals')
    plt.legend()
    plt.savefig('gdp_residuals.png')
    plt.clf()

    # Plot predictions
    boundmin = min(train_predictions) - 400
    boundmax = max(train_predictions) + 400
    plt.scatter([unstandardize(train['gdp'], train_labels[i]) for i in range(len(train_labels))], train_predictions)
    plt.xlabel('True Values (Billion $USD)')
    plt.ylabel('Predictions (Billion $USD)')
    plt.title('Predicted vs. True')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([boundmin, boundmax])
    plt.ylim([boundmin, boundmax])
    plt.savefig('gdp_predicted_v_true.png')
    plt.clf()

    return dec_mae, gdp_mae, train_resid_abs_mean, dev_resid_abs_mean, len(history.history['loss'])


def parse_conv_layers(conv_layers):
    """
    Parse convolutional layers argument.

    Returns tuple (kernel_size, filters, pool_size) for each layer.
    """
    conv_params = []
    for layer in conv_layers.split(','):
        params = [int(p) for p in layer.split('-')]
        if len(params) < 2:
            raise ValueError('-C must be [kernel_size]-[filters]-[pool_size (optional)],[filters]-...')
        elif len(params) == 2:
            params.append(2)
            conv_params.append(tuple(params))
        else:
            conv_params.append(tuple(params[: 3]))

    assert conv_params  # Make sure at least one was added
    return conv_params


def build_model(nhr, nreg, lr, wg, wd, conv_layers, l1, l2, lgdp):
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
    # moving along just one dimension.
    # Parameters for the convolutional layers are given as a comma-separated argument:
    #
    #     [kernel_size]-[filters]-[pool_size (optional)],[filters]-...
    #
    # These are fed directly into the Conv1D layer, which has parameters:
    #     Conv1D(filters, kernel_size, ...)
    conv_params = parse_conv_layers(conv_layers)

    i = 0
    for param_set in conv_params:
        if i == 0:
            convolutions = layers.Conv1D(param_set[1], param_set[0], padding='same', activation='relu')(input_numeric)
        else:
            convolutions = layers.Conv1D(param_set[1], param_set[0], padding='same', activation='relu')(convolutions)
        convolutions = layers.MaxPool1D(param_set[2])(convolutions)
        i += 1

    feature_layer = layers.Flatten()(convolutions)

    # Merge the inputs together and end our encoding with fully connected layers
    encoded = layers.Dense(l1, activation='relu')(feature_layer)
    encoded = layers.Dense(l2, activation='relu', name='FinalEncoding')(encoded)

    # At this point, the representation is the most encoded and small; now let's build the decoder
    decoded = layers.Dense(l1, activation='relu')(encoded)
    decoded = layers.Dense(convolutions.shape[1] * convolutions.shape[2], activation='relu')(decoded)

    decoded = layers.Reshape((convolutions.shape[1], convolutions.shape[2]), name='UnFlatten')(decoded)

    for param_set in reversed(conv_params):
        i -= 1
        decoded = layers.UpSampling1D(param_set[2])(decoded)
        if i == 0:
            decoded = layers.Conv1D(nreg, param_set[0], padding='same', activation='relu', name='DecoderOut')(decoded)
        else:
            decoded = layers.Conv1D(conv_params[i - 1][1], param_set[0], padding='same', activation='relu')(decoded)

    # Time since start input (for dealing with energy efficiency changes)
    input_time = layers.Input(shape=(1,), name='TimeSinceStart')

    # Natural gas data at a weekly level
    input_gas = layers.Input(shape=(1,), name='NaturalGas')

    # Petroleum data at a weekly level
    input_petrol = layers.Input(shape=(1,), name='Petroleum')

    # This is our actual output, the GDP prediction
    merged_layer = layers.Concatenate()([encoded, input_time, input_gas, input_petrol])
    output = layers.Dense(lgdp, activation='relu', name='OutputHiddenLayer')(merged_layer)
    output = layers.Dense(1, activation='linear', name='GDP_Output')(output)

    autoencoder = keras.models.Model([input_numeric, input_time, input_gas, input_petrol], [output, decoded])

    # Specify loss functions and weights for each output
    autoencoder.compile(loss={'GDP_Output': 'mean_squared_error', 'DecoderOut': 'mean_squared_error'},
                        loss_weights={'GDP_Output': wg, 'DecoderOut': wd},
                        optimizer=tf.keras.optimizers.RMSprop(lr=lr),
                        metrics=['mean_absolute_error', 'mean_squared_error'])
    return autoencoder


def run_prediction(model, dset, dset_name, labs):
    """
    Get predictions from the model with known results.

    Returns the predictions and the residuals (predicted - actual).
    """
    predictions = np.empty(len(dset['elec']))
    residuals = np.empty(len(dset['elec']))
    print(f'Predicting GDP with {dset_name} data')
    for i in range(len(dset['elec'])):
        pred = model.predict([dset['elec'][i], np.repeat(dset['time'][i], dset['elec'][i].shape[0]),
                              dset['gas'][i], dset['petrol'][i]], batch_size=1)[0]
        pred = unstandardize(dset['gdp'], pred.mean()).round(6)
        predictions[i] = pred
        residuals[i] = pred - unstandardize(dset['gdp'], labs[i])
        print(f'Predicted: ${pred}\tActual: ${unstandardize(dset["gdp"], labs[i])}')

    return (predictions, residuals)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('-lr', type=float, help='The learning rate (a float)', default=0.01)

    # CNN parameters
    cnn_help = 'Comma-separated list defining convolutional layers (kernel_length-filters) [default: "8-20,7-3-12"]'
    parser.add_argument('-C', type=str, help=cnn_help, default="8-20,7-3-12")

    # Hidden layers parameters
    hl1_help = 'The number of units in the first hidden layer (int) [default: 16]'
    hl2_help = 'The number of units in the final encoding (int) [default: 8]'
    hlg_help = 'The number of units in the hidden layer of the GDP branch (int) [default: 16]'
    parser.add_argument('-L1', type=int, help=hl1_help, default=16)
    parser.add_argument('-L2', type=int, help=hl2_help, default=8)
    parser.add_argument('-lgdp', type=int, help=hlg_help, default=16)

    # Loss function weights
    parser.add_argument('-wgdp', type=float,
                        help='Weight to apply to the loss of the GDP output [default: 0.5]',
                        default=0.5)
    parser.add_argument('-wdec', type=float,
                        help='Weight to apply to the loss of the decoder output [default: 0.5]',
                        default=0.5)

    # General model parameters
    parser.add_argument('-epochs', type=int, help='The number of epochs to train for', default=5000)
    parser.add_argument('-patience', type=int,
                        help='How many epochs to continue training without improving dev accuracy (int) [default: 50]',
                        default=50)
    parser.add_argument('-model', type=str,
                        help='Save the best model with this prefix, ignored if empty (string)',
                        default='')

    return parser.parse_args()


def run(args):
    """
    Get hyperparams, load and standardize data, train and evaluate.

    To see the model structure, see gdp_model.png.

    Given the general model structure, there are many parameters that can be
    adjusted for a given training. These hyperparameters can be tuned for best
    performance:

    lr - Optimization algorithm's learning rate
    C - Convolutional layers
    L1 - Hidden layer after convolutional layers
    L2 - Final encoded layer, represents features from electricity dataset
    lgdp - Hidden layer in GDP branch
    wgdp - Weight given to GDP output
    wdec - Weight given to decoder output
    epochs - Number of epochs before stopping
    patience - Number of epochs to stop after if no better result is found
    """
    st = time.time()

    # avoid clutter from old models / layers
    keras.backend.clear_session()

    # Set up diagnostics
    hyperparams = [k for k in vars(args).keys()]
    res_metrics = ['decoder_mae', 'GDP_mae', 'train_residuals_abs_mean', 'validation_residuals_abs_mean', 'nepoch']
    diag_file = DiagnosticFile('gdp_results.csv', hyperparams, res_metrics)

    # Load inputs
    train, dev = load_inputs(['train', 'dev'])

    # Run model
    results = train_model(train, dev, args, args.model, plots=True)

    # Record results
    notes = ''
    hpar_values = [v for k, v in vars(args).items()]
    diag_file.write(hpar_values, results, notes)

    print(f'Done in {time.time() - st} seconds.')


if __name__ == '__main__':
    # Hyperparameters passed in via command line
    run(get_args())
