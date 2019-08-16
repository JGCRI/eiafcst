"""
Convolutional neural net with auto encoder for predicting GDP from energy use.

Iteration 3
-----------
For our third attempt, we are using the residuals from the temperature
and electricity model, as well as natural gas and petroleum inputs. In
addition, we are inputting the GDP from the previous quarter. The goal
is to predict quarterly GDP values from these inputs, using a convolutional
neural network with an auto-encoder.

Caleb Braun
6/14/19
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
    val = val * gdp_orig.std() + gdp_orig.mean()
    if isinstance(val, float):
        return round(val, 2)
    else:
        return val.round(2)


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
        inputs.append({f.replace('.npy', ''): np.load(os.path.join(input_dir, f), allow_pickle=True)
                       for f in dset_files})

    return inputs


def batch_generator(ele_residuals, timestep, gdp_prev, gas_arr,
                    pet_arr, input_switch, encoder_arr, labels):
    """
    Generate a batch for training or evaluating the model.

    The input arrays have 2 dimensions of variable size:
     1. The number of cases (quarters)
     2. The number of weeks in a case
    The data therefore cannot be represented as a numpy array because not
    all dimensions match. To work around this, the data is provided from a
    generator where each batch is one case, so the input array has a fixed
    number of weeks (although this number is variable in each batch) and can
    be represented in a 3d numpy array.
    """
    i = 0
    while True:
        # print(f'Input length: {len(elec_residuals)}\tBATCH #{i % len(elec_residuals)}')
        ele = ele_residuals[i]
        gas = gas_arr[i]
        pet = pet_arr[i]

        nwk = ele.shape[0]
        ts = np.arange(timestep[i], timestep[i] + 1, 1 / nwk)
        gdp = np.repeat(gdp_prev[i], nwk)
        labs = np.repeat(labels[i], nwk)

        yield ([ele, ts, gdp, gas, pet, input_switch, encoder_arr], [labs, ele])
        i = (i + 1) % len(ele_residuals)


def train_model(train, dev, hpars, save_best, plots=True):
    """
    Run the deep learning model.

    Trains and evaluates the model. Assumes all inputs have been verified at
    this point.

    :param train:       Dictionary of all training datasets for all inputs and outputs.
    :param dev:         Dictionary of all validation datasets for all inputs and outputs.
    :param hpars:       Dictionary of all hyperparameters.
    :param save_best:   Location to store best model; if empty, does not save.
    :param plots:       Boolean; generate plots of model performance? [default: True]
    """
    # -----------------------------------------------------------------------
    # Prepare data for training.
    # -----------------------------------------------------------------------

    # standardize GDP values (elec values already standardized from prev. model)
    train_labels = (train['gdp'] - train['gdp'].mean()) / train['gdp'].std()
    dev_labels = (dev['gdp'] - dev['gdp'].mean()) / dev['gdp'].std()

    # Number of hours (168), aggregate electricity regions (14), and cases
    nhrweek = train['elec'][0].shape[1]
    nregion = train['elec'][0].shape[2]
    assert nhrweek == 168

    # -----------------------------------------------------------------------
    # Set up batching and callbacks for training.
    # -----------------------------------------------------------------------
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=hpars.patience,
                                               restore_best_weights=True)   # Important for keeping best model

    ## Set up the fixed encoder inputs.  We are training here, so we
    ## will want input_switch == 0.  The width of the encoder inputs
    ## is equal to hpars.L2; its values are arbitrary, since the
    ## input_switch will mask it out.
    input_switch = np.zeros(1)
    input_encoder = np.zeros(hpars.L2)
    
    # Batches are not all equally sized, so they need to be generated on the fly
    train_generator = batch_generator(train['elec'], train['time'], train['gdp_prev'],
                                      train['gas'], train['petrol'],
                                      input_switch, input_encoder, train_labels)
    dev_generator = batch_generator(dev['elec'], dev['time'], dev['gdp_prev'],
                                    dev['gas'], dev['petrol'],
                                    input_switch, input_encoder, dev_labels)

    # -----------------------------------------------------------------------
    # Build the model
    # -----------------------------------------------------------------------
    gdp_out_name = 'GDP_Output'
    dec_out_name = 'DecoderOut'

    model = build_model(nhrweek, nregion, hpars.C, hpars.L1, hpars.L2,
                        hpars.lgdp1, hpars.lgdp2, gdp_out_name, dec_out_name)

    if plots:
        plot_model(model, to_file='gdp_model.png', show_shapes=True, show_layer_names=True)

    # -----------------------------------------------------------------------
    # Train first the autoencoder, then freeze it and train the GDP branch
    # -----------------------------------------------------------------------
    # Freeze GDP layers (just for efficiency)
    for i, layer in enumerate(model.layers):
        layer.trainable = not layer.name.startswith('GDP_')  # There's probably a better way to do this

    model = compile_model(model, gdp_out_name, dec_out_name, 0.0, 1.0, hpars.lr)
    model.summary()

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train['elec']),
                                  epochs=hpars.epochs,
                                  verbose=0,
                                  callbacks=[early_stop],
                                  validation_data=dev_generator,
                                  validation_steps=len(dev['elec']))

    if plots:
        plot_history(history,
                     cols=[f'{dec_out_name}_mean_absolute_error', f'val_{dec_out_name}_mean_absolute_error'],
                     labs=['Train Decoder Error', 'Val Decoder Error', 'Train GDP Error', 'Val GDP Error'],
                     savefile='dec_train_history.png')

    # Freeze non-GDP layers
    for i, layer in enumerate(model.layers):
        layer.trainable = layer.name.startswith('GDP_')  # There's probably a better way to do this

    model = compile_model(model, gdp_out_name, dec_out_name, 1.0, 0.0, hpars.lr * 10)  # does better with higher lr
    model.summary()

    # Now do the GDP training
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train['elec']),
                                  epochs=hpars.epochs,
                                  verbose=0,
                                  callbacks=[early_stop],
                                  validation_data=dev_generator,
                                  validation_steps=len(dev['elec']))

    if plots:
        plot_history(history,
                     cols=[f'{gdp_out_name}_mean_absolute_error', f'val_{gdp_out_name}_mean_absolute_error'],
                     labs=['Train Decoder Error', 'Val Decoder Error', 'Train GDP Error', 'Val GDP Error'],
                     savefile='gdp_train_history.png')

    if save_best:
        model.save(save_best)

    # -----------------------------------------------------------------------
    # Model evaluation
    # -----------------------------------------------------------------------

    # Evaluate the model on our validation set
    
    dev_metrics = model.evaluate_generator(generator=dev_generator, steps=len(dev['elec']))

    # Metrics are specified in compile_model
    dev_gdp_mae = dev_metrics[3]
    dev_dec_mae = dev_metrics[5]

    train_predictions, train_residuals = run_prediction(model, train, 'training', train_labels)
    dev_predictions, dev_residuals = run_prediction(model, dev, 'development', dev_labels)

    dev_resid_abs_mean = np.abs(dev_residuals).mean()
    train_resid_abs_mean = np.abs(train_residuals).mean()
    print(f"Validation set residuals absolute mean {dev_resid_abs_mean}")
    print(f"Training set residuals absolute mean {train_resid_abs_mean}")

    if plots:
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
        plt.scatter(train['gdp'], train_predictions)
        plt.scatter(dev['gdp'], dev_predictions)
        plt.xlabel('True Values (Billion $USD)')
        plt.ylabel('Predictions (Billion $USD)')
        plt.title('Predicted vs. True')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([boundmin, boundmax])
        plt.ylim([boundmin, boundmax])
        plt.savefig('gdp_predicted_v_true.png')
        plt.clf()

        # Predictions timeline
        timex = np.concatenate([train['time'], dev['time']])
        order = np.argsort(timex)
        predsy = np.concatenate([train_predictions, dev_predictions])[order]
        truesy = np.concatenate([train['gdp'], dev['gdp']])[order]
        plt.plot(timex[order], truesy, label='True Values')
        plt.plot(timex[order], predsy, label='Predictions')
        plt.scatter(dev['time'], dev_predictions, label='Dev set predictions')
        plt.xlabel('Quarters since 2006Q1')
        plt.ylabel('Billion $USD')
        plt.legend()
        plt.show()

    return dev_dec_mae, dev_gdp_mae, train_resid_abs_mean, dev_resid_abs_mean, len(history.history['loss'])


def parse_conv_layers(conv_layers):
    """
    Parse convolutional layers argument.

    Returns tuple (kernel_size, filters, pool_size) for each layer. Pool size
    is optional and will be given a default value of 2 if not specified.
    """
    conv_params = []
    for layer in conv_layers.split(','):
        params = [int(p) for p in layer.split('-')]
        if len(params) < 2 or 3 < len(params):
            raise ValueError('-C must be [kernel_size]-[filters]-[pool_size (optional)],[filters]-...')
        elif len(params) == 2:
            params.append(2)
            conv_params.append(tuple(params))
        else:
            conv_params.append(tuple(params))  # All 3 options provided

    assert conv_params  # Make sure at least one was added
    return conv_params


def build_model(nhr, nreg, conv_layers, l1, l2, lgdp1, lgdp2, gdp_out_name, dec_out_name):
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

    # Time since start input (for dealing with energy efficiency changes)
    input_time = layers.Input(shape=(1,), name='TimeSinceStart')

    # Previous quarter's GDP
    input_gdp_prev = layers.Input(shape=(1,), name='GDPPrev')

    # Natural gas data at a weekly level
    input_gas = layers.Input(shape=(1,), name='NaturalGas')

    # Petroleum data at a weekly level
    input_petrol = layers.Input(shape=(1,), name='Petroleum')

    ## Add a way to specify specific encodings to investigate what the
    ## encoder is responding to.  The following scalar is either 1 or
    ## zero.  0 means normal operation; 1 means we use a specified
    ## input for the encoding.
    input_switch = layers.Input(shape=(1,), name='EncoderSwitch')
    encoder_input = layers.Input(shape=(l2,), name='EncoderInput') # ignored if input_switch == 0.

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
            convolutions = layers.Conv1D(param_set[1], param_set[0], padding='same',
                                         activation='relu', bias_initializer='glorot_uniform')(input_numeric)
        else:
            convolutions = layers.Conv1D(param_set[1], param_set[0], padding='same',
                                         activation='relu', bias_initializer='glorot_uniform')(convolutions)
        convolutions = layers.MaxPool1D(param_set[2])(convolutions)
        i += 1

    feature_layer = layers.Flatten()(convolutions)

    # Merge the inputs together and end our encoding with fully connected layers
    encoded = layers.Dense(l1, bias_initializer='glorot_uniform')(feature_layer)
    encoded = layers.LeakyReLU()(encoded)
    encoded = layers.Dense(l2, bias_initializer='glorot_uniform', name='FinalEncoding')(encoded)
    encoded = layers.LeakyReLU()(encoded)

    ## Implement the input switch (see above).
    def iswitch(input_switch, encoder_in, encoder_out):
        one = keras.backend.ones(shape=(1,))
        input_switch_complement = layers.subtract([one, input_switch])
        
        encoder_out = layers.multiply([encoder_out,input_switch_complement])
        encoder_in = layers.multiply([encoder_in, input_switch])
        encoded_out = layers.add([encoder_out, encoder_in], name='SwitchedEncoding')

    encoded = layers.Lambda(iswitch)(input_switch, encoder_input, encoded) 
        
    
    # At this point, the representation is the most encoded and small; now let's build the decoder
    decoded = layers.Dense(l1, bias_initializer='glorot_uniform')(encoded)
    decoded = layers.LeakyReLU()(decoded)
    decoded = layers.Dense(convolutions.shape[1] * convolutions.shape[2], bias_initializer='glorot_uniform')(decoded)
    decoded = layers.LeakyReLU()(decoded)

    decoded = layers.Reshape((convolutions.shape[1], convolutions.shape[2]), name='UnFlatten')(decoded)

    for param_set in reversed(conv_params):
        i -= 1
        decoded = layers.UpSampling1D(param_set[2])(decoded)
        if i == 0:
            decoded = layers.Conv1D(nreg, param_set[0], padding='same', activation='linear',
                                    bias_initializer='glorot_uniform', name=dec_out_name)(decoded)
        else:
            decoded = layers.Conv1D(conv_params[i - 1][1], param_set[0], padding='same',
                                    activation='relu', bias_initializer='glorot_uniform')(decoded)


    # This is our actual output, the GDP prediction
    merged_layer = layers.Concatenate()([encoded, input_time, input_gdp_prev, input_gas, input_petrol])

    gdp_hidden_layer = layers.Dense(lgdp1, name='GDP_Hidden')(merged_layer)
    gdp_hidden_layer = layers.LeakyReLU()(gdp_hidden_layer)
    if lgdp2 > 0:
        gdp_hidden_layer = layers.Dense(lgdp2, name='GDP_Hidden2',
                                        kernel_regularizer=keras.regularizers.l1(0))(gdp_hidden_layer)
        gdp_hidden_layer = layers.LeakyReLU()(gdp_hidden_layer)

    output = layers.Dense(1, activation='linear', name=gdp_out_name)(gdp_hidden_layer)

    autoencoder = keras.models.Model(inputs=[input_numeric, input_time, input_gdp_prev, input_gas, input_petrol],
                                     outputs=[output, decoded])

    return autoencoder


def compile_model(model, o1, o2, w1, w2, lr):
    """
    Compile a model.

    :param o1:  Name of the first output
    :param o2:  Name of the second output
    :param w1:  Loss weight of the first output
    :param w2:  Loss weight of the second output
    :param lr:  Learning rate
    """
    # Specify loss functions and weights for each output
    model.compile(loss={o1: 'mean_squared_error', o2: 'mean_squared_error'},
                  loss_weights={o1: w1, o2: w2},
                  optimizer=tf.keras.optimizers.RMSprop(lr=lr),
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def run_prediction(model, dset, dset_name, labs, normal_mode=True):
    """
    Get predictions from the model with known results.

    Returns the predictions and the residuals (predicted - actual).

    If normal mode is True, run electricity inputs through the
    encoder; if false, submit a specified encoding.
    """
    predictions = np.empty(len(dset['elec']))
    residuals = np.empty(len(dset['elec']))
    decoder_outs = np.empty(dset['elec'].shape)
    encoder_input_shape = model.get_layer(name='EncoderSwitch').input_shape
    encoder_input_shape[0] = len(dset['elec'])
    if normal_mode:
        input_switch = np.zeros(len(dset['elec']))
        encoder_input = np.zeros(encoder_input_shape)
        print(f'Predicting GDP with {dset_name} data')
    else:
        input_switch = np.ones(len(dset['elec']))
        print(f'Running encoding interpretability tests')
        encoder_input = np.zeros(encoder_input_shape)
        nrow = len(dset['elec'])
        for i in range(nrow):
            ncol = encoder_input_shape[1]
            idx1 = i % ncol
            idx2 = int(i/ncol)
            row = np.zeros(ncol)
            row[idx1,:] = idx2
            
        
    for i in range(len(dset['elec'])):
        ele = dset['elec'][i]
        gas = dset['gas'][i]
        pet = dset['petrol'][i]
        enc_input = encoder_input[i]

        nwk = ele.shape[0]
        ts = np.arange(dset['time'][i], dset['time'][i] + 1, 1 / nwk)
        gdp_prev = np.repeat(dset['gdp_prev'][i], nwk)

        preds = model.predict([ele, ts, gdp_prev, gas, pet, input_switch, enc_input], batch_size=1)
        pred = preds[0]
        pred = unstandardize(dset['gdp'], pred.mean()).round(6)
        decoder_outs[i] = preds[1]
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
    lgdp1_help = 'The number of units in the first hidden layer of the GDP branch (int) [default: 4]'
    lgdp2_help = 'The number of units in the second hidden layer of the GDP branch (int) [default: 4]'
    parser.add_argument('-L1', type=int, help=hl1_help, default=16)
    parser.add_argument('-L2', type=int, help=hl2_help, default=8)
    parser.add_argument('-lgdp1', type=int, help=lgdp1_help, default=4)
    parser.add_argument('-lgdp2', type=int, help=lgdp2_help, default=4)

    # Loss function weights
    parser.add_argument('-wgdp', type=float,
                        help='Weight to apply to the loss of the GDP output [default: 0.0]',
                        default=0.0)
    parser.add_argument('-wdec', type=float,
                        help='Weight to apply to the loss of the decoder output [default: 1.0]',
                        default=1.0)

    # General model parameters
    parser.add_argument('-epochs', type=int, help='The number of epochs to train for', default=8000)
    parser.add_argument('-patience', type=int,
                        help='How many epochs to continue training without improving dev accuracy (int) [default: 50]',
                        default=50)
    parser.add_argument('-model', type=str,
                        help='Save the best model with this prefix, ignored if empty (string)',
                        default='')

    return parser.parse_args()


def run(args, diag_fname='gdp_2hidden.csv'):
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
    lgdp1 - First hidden layer in GDP branch
    lgdp2 - Second hidden layer in GDP branch
    epochs - Number of epochs before stopping
    patience - Number of epochs to stop after if no better result is found
    """
    st = time.time()

    # avoid clutter from old models / layers
    keras.backend.clear_session()

    # Set up diagnostics
    hyperparams = [k for k in vars(args).keys()]
    res_metrics = ['decoder_mae', 'GDP_mae', 'train_residuals_abs_mean', 'validation_residuals_abs_mean', 'nepoch']
    diag_file = DiagnosticFile(diag_fname, hyperparams, res_metrics)

    # Load inputs
    train, dev = load_inputs(['train', 'dev'])

    # Run model
    results = train_model(train, dev, args, args.model, plots=True)

    # Record results
    notes = 'added second hidden layer'
    time_taken = int(time.time() - st)
    hpar_values = [v for k, v in vars(args).items()]
    diag_file.write(hpar_values, results, time_taken, notes)

    print(f'Done in {time.time() - st} seconds.')


if __name__ == '__main__':
    # Hyperparameters passed in via command line
    run(get_args())
