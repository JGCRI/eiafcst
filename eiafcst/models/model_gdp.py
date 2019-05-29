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

from eiafcst.dataprep import FuelParser
from eiafcst.dataprep.utils import read_training_data, plot_history, DiagnosticFile, add_quarter_and_week
from eiafcst.dataprep.economic import parse_gdp

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model


def unstandardize(gdp_stats, val):
    """Convert standardized GDP to dollar values."""
    return round(val * gdp_stats['std'] + gdp_stats['mean'], 2)


def load_fuel(fp, fuel_type, years, train_idx, valid_idx, test_idx):
    """
    Prepare fuel data as input to the GDP model.

    Filters out categories like residential consumption and electric power, as
    those categories are presumably just repeating information provided by the
    electricity inputs.

    :param fp:      FuelParser object
    :param years:   Sequence of years to filter gas data to
    """
    fuel = fp.parse(fuel_type)

    fuel = fuel[fuel['EconYear'].isin(years)]

    val_col = fuel.columns[-1]
    var_col = fuel.columns[-2]

    # Filter to best GDP categories for natural gas
    if fuel_type == 'gas':
        gdp_categories = ['Industrial Consumption', 'Lease and Plant Fuel Consumption', 'Pipeline & Distrubtion Use']
        fuel = fuel[fuel[var_col].isin(gdp_categories)]

    # Aggregate categories to weekly total
    fuel = fuel.groupby(['EconYear', 'quarter', 'week'], as_index=False).agg({val_col: 'sum'})

    # Standardize
    fuel[val_col] = (fuel[val_col] - fuel[val_col].mean()) / fuel[val_col].std()

    # Add case (quarter) number
    fuel['case'] = (fuel['EconYear'] - fuel['EconYear'].min()) * 4 + fuel['quarter'] - 1

    train = [fuel.loc[fuel['case'] == i, val_col].values for i in train_idx]
    valid = [fuel.loc[fuel['case'] == i, val_col].values for i in valid_idx]
    test = [fuel.loc[fuel['case'] == i, val_col].values for i in test_idx]

    return train, valid, test


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


def run(trainx, trainy, lr, wg, wd, conv_layers, l1, l2, lgdp, epochs, patience, model, plots=True):
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
    # trainx = '/Users/brau074/Documents/EIA/eiafcst/eiafcst/models/electricity/elec_model5_residuals.csv'

    # trainx is reshaped into list of arrays of quarterly vals [weeks, hours, regions]
    values, labels, gdp_stats = prep_data(trainx)
    # standardize GDP values (elec values already standardized from prev. model)
    labels['gdp'] = (labels['gdp'] - gdp_stats['mean']) / gdp_stats['std']

    # Number of hours (168), aggregate electricity regions (14), and cases
    nhrweek = values[0].shape[1]
    nregion = values[0].shape[2]
    assert nhrweek == 168
    ncases = len(values)

    # Randomly choose which cases are for training the model, which are for
    # model validation, and which are set aside for testing. The test set
    # and validation sets each contain 1/5 of all cases. Therefore the validation
    # set is 1/4 the size of the training set.
    np.random.seed(42)
    rand_idx = np.random.permutation(ncases)  # 27, 40, 26, 43
    train_idx, valid_idx, test_idx = np.split(rand_idx, [int(ncases * (3 / 5)), int(ncases * (4 / 5))])

    trainx = values[train_idx]
    validx = values[valid_idx]
    # testx = values[test_idx]
    trainy = labels.gdp.values[train_idx]
    validy = labels.gdp.values[valid_idx]
    # testy = labels.gdp.values[test_idx]
    train_timesteps = train_idx
    valid_timesteps = valid_idx

    # Separate the natural gas and petroleum data into cases matching the gdp
    # and electricity inputs.
    fp = FuelParser.FuelParser()
    yrs = labels['EconYear'].unique()
    train_gas, valid_gas, test_gas = load_fuel(fp, 'gas', yrs, train_idx, valid_idx, test_idx)
    train_petrol, valid_petrol, test_petrol = load_fuel(fp, 'petrol', yrs, train_idx, valid_idx, test_idx)

    model = build_model(nhrweek, nregion, lr, wg, wd, conv_layers, l1, l2, lgdp)

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
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit_generator(generator=batch_generator(trainx, train_timesteps, train_gas, train_petrol, trainy),
                                  steps_per_epoch=len(trainx),
                                  epochs=epochs,
                                  verbose=0,
                                  callbacks=[early_stop],
                                  validation_data=batch_generator(
                                      validx, valid_timesteps, valid_gas, valid_petrol, validy),
                                  validation_steps=len(validx))

    plot_history(history,
                 cols=['DecoderOutput_mean_absolute_error', 'val_DecoderOutput_mean_absolute_error',
                       'GDP_Output_mean_absolute_error', 'val_GDP_Output_mean_absolute_error'],
                 labs=['Train Decoder Error', 'Val Decoder Error', 'Train GDP Error', 'Val GDP Error'],
                 savefile='gdp_train_history.png')

    # Evaluate the model on our validation set
    metrics = model.evaluate_generator(generator=batch_generator(validx, valid_timesteps, valid_gas, valid_petrol, validy),
                                       steps=len(validx))

    gdp_mae = metrics[3]
    dec_mae = metrics[5]

    train_predictions = np.empty(len(trainx))
    train_residuals = np.empty(len(trainx))
    print('Predicting GDP with training data')
    for i in range(len(trainx)):
        train_pred = model.predict([trainx[i], np.repeat(train_timesteps[i], trainx[i].shape[0]),
                                    train_gas[i], train_petrol[i]], batch_size=1)[0]
        train_pred = unstandardize(gdp_stats, train_pred.mean()).round(6)
        train_predictions[i] = train_pred
        train_residuals[i] = train_pred - unstandardize(gdp_stats, trainy[i])
        print(f'Predicted: ${train_pred}\tActual: ${unstandardize(gdp_stats, trainy[i])}')

    print('Predicting GDP with validation data')
    valid_predictions = np.empty(len(validx))
    valid_residuals = np.empty(len(validx))
    for i in range(len(validx)):
        valid_pred = model.predict([validx[i], np.repeat(valid_idx[i], validx[i].shape[0]),
                                    valid_gas[i], valid_petrol[i]], batch_size=1)[0]
        valid_pred = unstandardize(gdp_stats, valid_pred.mean()).round(6)
        valid_predictions[i] = valid_pred
        valid_residuals[i] = valid_pred - unstandardize(gdp_stats, validy[i])
        print(f'Predicted: ${valid_pred}\tActual: ${unstandardize(gdp_stats, validy[i])}')

    valid_resid_abs_mean = np.abs(valid_residuals).mean()
    train_resid_abs_mean = np.abs(train_residuals).mean()
    print(f"Validation set residuals absolute mean {valid_resid_abs_mean}")
    print(f"Training set residuals absolute mean {train_resid_abs_mean}")

    plt.xlabel('Timestep')
    plt.ylabel('Prediction residual (Billion USD)')
    plt.scatter(valid_idx, valid_residuals, c='#ef8a62', label='validation')
    plt.scatter(train_idx, train_residuals, c='#67a9cf', label='train')
    plt.title('Residuals')
    plt.legend()
    plt.show()
    plt.clf()

    # Plot predictions
    boundmin = min(train_predictions) - 400
    boundmax = max(train_predictions) + 400
    plt.scatter([unstandardize(gdp_stats, trainy[i]) for i in range(len(trainy))], train_predictions)
    plt.xlabel('True Values (Billion $USD)')
    plt.ylabel('Predictions (Billion $USD)')
    plt.title('Predicted vs. True')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([boundmin, boundmax])
    plt.ylim([boundmin, boundmax])
    plt.show()
    plt.clf()

    return dec_mae, gdp_mae, train_resid_abs_mean, valid_resid_abs_mean, len(history.history['loss'])


def parse_conv_layers(conv_layers):
    """
    Parse convolutional layers argument.

    Returns tuple (filters, kernel_size, pool_size) for each layer.
    """
    conv_params = []
    for layer in conv_layers.split(','):
        params = [int(p) for p in layer.split('-')]
        if len(params) < 2:
            raise ValueError('-C must be [filters]-[kernel_size]-[pool_size (optional)],[filters]-...')
        elif len(params) == 2:
            params.append(2)
            conv_params.append(tuple(params))
        else:
            conv_params.append(tuple(params[:3]))

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
    #     [filters]-[kernel_size]-[pool_size (optional)],[filters]-...
    #
    # These are fed directly into the Conv1D layer, which has parameters:
    #     Conv1D(filters, kernel_size, ...)
    conv_params = parse_conv_layers(conv_layers)
    last_conv_filters, last_conv_length = conv_params[1][:2]

    convolutions = layers.Conv1D(conv_params[0][1], conv_params[0][0], padding='same', activation='relu')(input_numeric)
    convolutions = layers.MaxPool1D(conv_params[0][2])(convolutions)

    for param_set in conv_params[1:]:
        convolutions = layers.Conv1D(param_set[1], param_set[0], padding='same', activation='relu')(convolutions)
        convolutions = layers.MaxPool1D(param_set[2])(convolutions)

    feature_layer = layers.Flatten()(convolutions)

    # Merge the inputs together and end our encoding with fully connected layers
    encoded = layers.Dense(l1, activation='relu')(feature_layer)
    encoded = layers.Dense(l2, activation='relu', name='FinalEncoding')(encoded)

    # At this point, the representation is the most encoded and small
    # Now let's build the decoder
    decoded = layers.Dense(l1, activation='relu')(encoded)
    decoded = layers.Dense(last_conv_filters * last_conv_length, activation='relu')(decoded)

    decoded = layers.Reshape((last_conv_filters, last_conv_length), name='UnFlatten')(decoded)

    for param_set in reversed(conv_params):
        decoded = layers.Conv1D(param_set[1], param_set[0], padding='same', activation='relu')(decoded)
        decoded = layers.UpSampling1D(param_set[2])(decoded)

    decoded = layers.Conv1D(nreg, conv_params[0][0], padding='same', activation='relu', name='DecoderOutput')(decoded)

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
    autoencoder.compile(loss={'GDP_Output': 'mean_squared_error', 'DecoderOutput': 'mean_squared_error'},
                        loss_weights={'GDP_Output': wg, 'DecoderOutput': wd},
                        optimizer=tf.keras.optimizers.RMSprop(lr=lr),
                        metrics=['mean_absolute_error', 'mean_squared_error'])
    return autoencoder


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('trainx', type=str, help='Training dataset of electricity residuals')
    parser.add_argument('trainy', type=str, help='Training dataset of quarterly GDP values')
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


def main():
    """
    Get hyperparams, load and standardize data, train and evaluate.

    To see the model structure, see gdp_model.png.

    Given the general model structure, there are many parameters that can be
    adjusted for a given training. These hyperparameters can be tuned for best
    performance:

    trainx - File path to training dataset values
    trainy - File path to training dataset labels
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

    # Hyperparameters passed in via command line
    args = get_args()

    # Set up diagnostics
    hyperparams = [k for k in vars(args).keys()]
    res_metrics = ['decoder_mae', 'GDP_mae', 'train_residuals_abs_mean', 'validation_residuals_abs_mean', 'nepoch']
    diag_file = DiagnosticFile('gdp_results.csv', hyperparams, res_metrics)

    notes = ''

    # Run model
    results = run(args.trainx, args.trainy, args.lr, args.wgdp, args.wdec, args.C, args.L1,
                  args.L2, args.lgdp, args.epochs, args.patience, args.model, plots=True)

    # Record results
    hyper_values = [str(v) for k, v in vars(args).items()]
    diag_file.write(hyper_values, results, notes)

    print(f'Done in {time.time() - st} seconds.')


if __name__ == '__main__':
    main()
