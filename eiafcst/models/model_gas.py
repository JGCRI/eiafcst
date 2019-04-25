"""
Regression neural net predicting natural gas usage from temperature.

Caleb Braun
4/22/19
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

PLOTDIR = resource_filename('eiafcst', os.path.join('models', 'diagnostic'))
# INPUTDIR = '/Users/brau074/Documents/EIA/modeling/load/input'


class PrintDot(keras.callbacks.Callback):
    """Display training progress by printing a single dot for each completed epoch."""

    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


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


def plot_by_region(dataset):
    """
    Create density and line plots for each region.

    The parameter `dataset` should be have the three columns "ID", "Load (MW)"
    and "temp".
    """
    for reg in dataset.ID.unique():
        datareg = dataset[dataset['ID'] == reg]

        databin = datareg.copy()
        databin['temp'] = databin['temp'].round(1)

        sns.relplot(x='temp', y='Load (MW)', data=databin, ci="sd", kind='line')
        plt.title(reg)
        fig = plt.gcf()
        fig.set_size_inches(16, 10)
        plt.savefig(f'{PLOTDIR}/{reg}_line.png')

        sns.jointplot(x="temp", y="Load (MW)", data=datareg, kind="kde")
        plt.title(reg)
        plt.savefig(f'{PLOTDIR}/{reg}_density.png')
        plt.clf()


def prep_data(xpth, ypth, train_frac=0.8):
    """
    Combine gas and temperature datasets and prepare them for the model.

    :param xpth:        path to .pkl or .csv file containing temperature by hour
    :param ypth:        path to .pkl or .csv file containing natural gas
                        consumption by week
    :param train_frac:  fraction (number between 0-1) of data for training
    """
    HR_WK = 24 * 7  # hours in a week

    temperature = read_training_data(xpth).rename(columns={'ID': 'State'})
    gas = read_training_data(ypth)

    # Display dataset statistics that we'll use for standardization
    temp_stats = temperature[['State', 'temperature']].groupby('State').describe()
    gas_stats = gas[['State', 'mmcf']].groupby('State').describe()
    allstats = pd.concat([temp_stats, gas_stats], axis=1)
    print("\nInput dataset summary:")
    print(allstats.transpose(), "\n")

    temperature = add_quarter_and_week(temperature, 'time')

    # Remove temperature data for times not in gas data
    temperature = temperature.merge(gas, how='left', on=['EconYear', 'quarter', 'week', 'State'], validate='m:1')
    temperature = temperature[~(temperature.mmcf.isna())]
    temperature = temperature.sort_values(['State', 'EconYear', 'quarter', 'week'])

    assert len(gas) == len(temperature) / HR_WK

    # Our goal is an array with dimensions [weeks, hours, regions]
    nreg = len(temperature.State.unique())
    nweek = len(temperature) // (HR_WK * nreg)
    temp_arr = temperature.temperature.values.reshape((nreg, nweek, HR_WK))
    temp_arr = np.moveaxis(temp_arr, 0, 2)

    # Array with dimensions [weeks, regions]
    gas = gas.sort_values(['State', 'EconYear', 'quarter', 'week'])
    gas_arr = gas.mmcf.values.reshape((nreg, nweek))
    gas_arr = gas_arr.transpose()

    return temp_arr, gas_arr, allstats


def run(xpth, ypth, learning_rate, l1, l2, epochs, patience, checkpoint_path,
        embedding_size, sample_size=1000, plots=False):
    """
    Build and train the electricity model.

    Creates plots of process, if specified.
    """
    xpth = 'temperature_by_state_popweighted.csv'
    ypth = 'gas_weekly_by_state.csv'

    # Get the data as numpy arrays
    # temp_arr, gas_arr, allstats = prep_data(xpth, ypth)

    # For testing!!
    # np.save('temp_arr.npy', temp_arr)
    # np.save('gas_arr.npy', gas_arr)
    # allstats.to_pickle('allstats.pkl')
    temp_arr = np.load('temp_arr.npy')
    gas_arr = np.load('gas_arr.npy')
    allstats = pd.read_pickle('allstats.pkl')

    # Split the datasets into a training set and a test set. We will use the
    # test set in the final evaluation of our model.
    train_frac = 0.2
    nweek = temp_arr.shape[0]
    train_idx = np.random.choice(nweek, size=int(nweek * train_frac), replace=False)
    test_idx = np.setdiff1d(np.arange(nweek), train_idx)

    trainx = temp_arr[train_idx, :, :]
    trainy = gas_arr[train_idx, :]
    testx = temp_arr[test_idx, :, :]
    testy = gas_arr[test_idx, :]

    # Standardize all the data
    stdzd_train_x = standardize(trainx)
    stdzd_train_y = standardize(trainy)
    stdzd_test_x = standardize(testx)
    stdzd_test_y = standardize(testy)

    # The IDs cannot be used as inputs on their own, as the NN can only work
    # with meaningful numeric data. To address this, we use an Embedding
    # layer, which takes in the ID values represented as integers.
    category_names = allstats.index
    category_map = {name: id for id, name in enumerate(category_names)}

    train_region_embed_id = np.tile(np.arange(len(category_names)), stdzd_train_x.shape[0])
    test_region_embed_id = np.tile(np.arange(len(category_names)), stdzd_test_x.shape[0])

    stdzd_train_labels = stdzd_train_y.flatten()
    stdzd_test_labels = stdzd_test_y.flatten()

    input_layer_shape = stdzd_train_x.shape[1]  # Hours in a week

    # Converts to (case x 168) where every 51 rows is the case for all states for a given timestep
    train_weeks = np.moveaxis(stdzd_train_x, 2, 1).reshape(-1, input_layer_shape, 1)
    train_time = np.repeat(train_idx, len(category_names))
    test_weeks = np.moveaxis(stdzd_test_x, 2, 1).reshape(-1, input_layer_shape, 1)
    test_time = np.repeat(test_idx, len(category_names))

    # Build and show the model
    model = build_model(input_layer_shape, l1, l2, learning_rate, len(category_names), embedding_size)
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=0, period=5)

    history = model.fit(x=[train_weeks, train_time, train_region_embed_id],
                        y=stdzd_train_labels,
                        epochs=epochs, validation_split=0.2, verbose=0,
                        callbacks=[early_stop, PrintDot()])
    print()

    if plots:
        plot_history(history)
        plot_embeddings(model.layers[7].get_weights()[0], category_names)

    # Evaluate the model on our reserved test data
    loss, mae, mse = model.evaluate([test_weeks, test_time, test_region_embed_id],
                                    stdzd_test_labels, verbose=0)
    print("Testing set standardized Mean Abs Error: {:5.3f}".format(mae))

    stdzd_train_predictions = model.predict([train_weeks, train_time, train_region_embed_id]).flatten()
    stdzd_test_predictions = model.predict([test_weeks, test_time, test_region_embed_id]).flatten()

    trn_resid = stdzd_train_predictions - stdzd_train_labels
    tst_resid = stdzd_test_predictions - stdzd_test_labels

    if plots:
        plot_predictions(stdzd_test_labels, stdzd_test_predictions)
        plot_predictions_facet(model, category_names, allstats, stdzd_train_x, train_region_embed_id)
        plot_residuals(trn_resid, tst_resid, train_weeks.mean(axis=(1, 2)), test_weeks.mean(axis=(1, 2)))

    print("Test residuals mean {:5.4f}".format(np.mean(tst_resid)))
    print("Training residuals mean {:5.4f}".format(np.mean(trn_resid)))

    return mae, mse, round(np.mean(trn_resid), 5), round(np.mean(tst_resid), 5)


def standardize(x):
    """
    Standardize numeric features.

    It is good practice to standardize features that use different scales and
    ranges. Although the model might converge without feature standardization,
    it makes training more difficult, and it makes the resulting model dependent
    on the choice of units used in the input.

    :param x:           A 3d Numpy array
    """
    return (x - np.mean(x, axis=(0, 1))) / np.std(x, axis=(0, 1))


def unstandardize(z, allstats, vars):
    """Inverse of standardize."""
    for var in vars:
        stats = allstats[var]
        z = z.merge(stats[['mean', 'std']], left_on='ID', right_index=True)
        z.loc[:, var] = z[var] * z['std'] + z['mean']
        z = z.drop(columns=['mean', 'std'])

    return z


def build_model(input_layer_shape, l1, l2, lr, embed_in_dim, embed_out_dim):
    """
    We need three separate inputs, one for categorical data that gets an
    embedding, one for time since data start, and one for the numerical data.
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


def plot_residuals(trn_resid, tst_resid, stdzd_train_x, stdzd_test_x):
    """Plot residuals."""
    plt.xlabel('Standardized Temperature')
    plt.ylabel('Prediction residual')
    plt.scatter(stdzd_test_x, tst_resid, alpha=0.05)
    plt.scatter(stdzd_train_x, trn_resid, alpha=0.05)
    plt.savefig(f'{PLOTDIR}/residuals.png')
    plt.clf()


def plot_predictions_facet(model, category_names, allstats, stdzd_train_x, train_region_embed_id):
    """Facet plot of all temperature predictions."""
    temperature_seq = np.arange(-20, 40, 1) + 273.15
    npoints = len(temperature_seq)

    nrow = int(len(category_names)**0.5)
    ncol = int(len(category_names) / nrow + .999)

    plt.rcParams["figure.figsize"] = [20, 12]
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
    fig.suptitle('Modeled Natural Gas Use Predictions by Aggregate Region', fontsize=16)
    for i, region in enumerate(category_names):
        regmean = allstats.loc[region, 'temperature']['mean']
        regstd = allstats.loc[region, 'temperature']['std']
        stdzd_seq_data = (temperature_seq - regmean) / regstd
        stdzd_seq_data = np.repeat(stdzd_seq_data, 168).reshape(-1, 168, 1)
        timestep = np.zeros(npoints) + 1042

        stdzd_seq_predictions = model.predict([stdzd_seq_data, timestep, np.zeros(npoints) + i]).flatten()

        training_bounds = (allstats.loc[region, 'temperature'][['min', 'max']]).values
        training_bounds = (training_bounds - regmean) / regstd

        if len(category_names) == 1:
            ax = axes
        else:
            ax = axes[divmod(i, ncol)]
        ax.set_title(region)
        ax.axvspan(training_bounds[0], training_bounds[1], alpha=0.3, color='salmon')
        ax.set_xlim([-4, 4])
        ax.plot(stdzd_seq_data[:, 0, 0], stdzd_seq_predictions)

    fig.text(0.5, 0.04, 'Standardized Temperature', ha='center')
    fig.text(0.04, 0.5, 'Standardized Natural Gas COnsumption', va='center', rotation='vertical')
    plt.savefig(f'{PLOTDIR}/predictions.png')
    plt.clf()


def plot_predictions(labels, predictions):
    boundmin = min(min(labels), min(predictions))
    boundmax = max(max(labels), max(predictions))
    plt.scatter(labels, predictions, min(1, max(100 / len(labels), 0.01)))
    plt.xlabel('True Values [MW]')
    plt.ylabel('Predictions [MW]')
    plt.title('Predicted vs. True')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([boundmin, boundmax])
    plt.ylim([boundmin, boundmax])
    plt.savefig(f'{PLOTDIR}/predicted_v_true.png')
    plt.clf()


def plot_embeddings(embeddings, category_names):
    fig, ax = plt.subplots()
    ax.scatter(embeddings[:, 0], embeddings[:, 1])
    for i, label in enumerate(category_names):
        ax.annotate(label, (embeddings[i, 0], embeddings[i, 1]))
    plt.title("Embedding Values")
    plt.savefig(f'{PLOTDIR}/embeddings.png')
    plt.clf()


def plot_history(history):
    """
    Visualize the model's training progress using the stats stored in the
    history object.
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.title('Training Performance')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()

    # plt.figure()
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Square Error [$Load (MW)^2$]')
    # plt.plot(hist['epoch'], hist['mean_squared_error'],
    #          label='Train Error')
    # plt.plot(hist['epoch'], hist['val_mean_squared_error'],
    #          label='Val Error')
    # plt.ylim([0, 20])
    # plt.legend()
    plt.savefig(f'{PLOTDIR}/history.png')
    plt.clf()


def main():
    """Get hyperparams, load and standardize data, train and evaluate."""
    st = time.time()
    notes = ''

    args = get_args()

    diag_dir = resource_filename('eiafcst', os.path.join('models', 'diagnostic'))
    res_fname = os.path.join(diag_dir, 'gas_results.csv')

    # Record results always
    if not os.path.exists(res_fname):
        with open(res_fname, 'w') as results_file:
            results_file.write(','.join(['samplesize', 'lr', 'L1', 'L2', 'patience', 'embedsize', 'mae', 'mse',
                                         'mean train residual', 'mean test residual', 'notes']))
            results_file.write('\n')

    with open(res_fname, 'a') as outfile:
        results = run(args.trainx, args.trainy, args.lr, args.L1, args.L2, args.epochs, args.patience,
                      args.model, args.embedsize, plots=True)

        argline = ','.join(str(a) for a in ['all', args.lr, args.L1, args.L2, args.patience, args.embedsize])
        outfile.write(argline + ',' + ','.join([str(r) for r in results]) + ',' + notes)
        outfile.write('\n')

        print(f'Done in {time.time() - st} seconds.')


if __name__ == '__main__':
    main()
