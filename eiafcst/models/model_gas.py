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


def prep_data(trainx, trainy, train_frac=0.8):
    """
    Combine gas and temperature datasets and prepare them for the model.

    :param trainx:      path to .pkl or .csv file containing temperature by hour
    :param trainy:      path to .pkl or .csv file containing natural gas
                        consumption by week
    :param train_frac:  fraction (number between 0-1) of data for training
    """
    HR_WK = 24 * 7  # hours in a week

    temperature = read_training_data(trainx)
    gas = read_training_data(trainy)

    temperature = temperature.rename(columns={'ID': 'State'})
    temperature = add_quarter_and_week(temperature, 'time')

    # Remove temperature data for times not in gas data
    temperature = temperature.merge(gas, how='left', on=['EconYear', 'quarter', 'week', 'State'], validate='m:1')
    temperature = temperature[~(temperature.value.isna())]
    temperature = temperature.sort_values(['State', 'EconYear', 'quarter', 'week'])

    assert len(gas) == len(temperature) / HR_WK

    # Array with dimensions [regions, weeks, hours]
    nreg = len(temperature.State.unique())
    nweek = len(temperature) // (HR_WK * nreg)
    temp_arr = temperature.temperature.values.reshape((nreg, nweek, HR_WK))

    # Array with dimensions [regions, weeks]
    gas = gas.sort_values(['State', 'EconYear', 'quarter', 'week'])
    gas_arr = gas.value.values.reshape((nreg, nweek))

    # Split the datasets into a training set and a test set. We will use the
    # test set in the final evaluation of our model.
    train_idx = np.random.choice(nweek, size=int(nweek * train_frac), replace=False)
    test_idx = np.setdiff1d(np.arange(nweek), train_idx)
    temp_arr[:, train_idx, :].shape
    temp_arr[:, test_idx, :].shape

    allstats = dataset.groupby('ID').describe()
    print("\nInput dataset summary:")
    print(allstats.transpose(), "\n")

    return train_dataset, test_dataset, allstats


def run(trainx, trainy, learning_rate, l1, l2, epochs, patience, checkpoint_path,
        embedding_size, sample_size=1000, plots=False):
    """
    Build and train the electricity model.

    Creates plots of process, if specified.
    """
    # Split data into training and test datasets
    train_dataset, test_dataset, allstats = prep_data(trainx, trainy)

    # Standardize all the data
    stdzd_train_data = standardize(train_dataset, allstats, ['Load (MW)', 'temp'])
    stdzd_test_data = standardize(test_dataset, allstats, ['Load (MW)', 'temp'])

    # Separate the target value, or "label", from the features. This label is
    # the value that we will train the model to predict (electric load).
    stdzd_train_labels = stdzd_train_data.pop('Load (MW)')
    stdzd_test_labels = stdzd_test_data.pop('Load (MW)')

    # The IDs cannot be used as inputs on their own, as the NN can only work
    # with meaningful numeric data. To address this, we use an Embedding
    # layer, which takes in the ID values represented as integers.
    category_names = stdzd_train_data['ID'].unique()
    category_map = {name: id for id, name in enumerate(category_names)}
    train_region_embed_id = stdzd_train_data.pop('ID').map(category_map)
    test_region_embed_id = stdzd_test_data.pop('ID').map(category_map)

    # Build and show the model
    model = build_model(l1, l2, learning_rate, len(category_names), embedding_size)
    model.summary()

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=0, period=5)

    history = model.fit(x=[stdzd_train_data, train_region_embed_id],
                        y=stdzd_train_labels,
                        epochs=epochs, validation_split=0.2, verbose=0,
                        callbacks=[early_stop, PrintDot()])
    print()

    if plots:
        plot_history(history)
        plot_embeddings(model.layers[1].get_weights()[0], category_names)

    # Evaluate the model on our reserved test data
    loss, mae, mse = model.evaluate([stdzd_test_data, test_region_embed_id],
                                    stdzd_test_labels, verbose=0)
    print("Testing set standardized Mean Abs Error: {:5.3f}".format(mae))

    stdzd_train_predictions = model.predict([stdzd_train_data, train_region_embed_id]).flatten()
    stdzd_test_predictions = model.predict([stdzd_test_data, test_region_embed_id]).flatten()

    trn_resid = stdzd_train_predictions - stdzd_train_labels
    tst_resid = stdzd_test_predictions - stdzd_test_labels

    if plots:
        plot_predictions(stdzd_test_labels, stdzd_test_predictions)
        plot_predictions_facet(model, category_names, allstats, stdzd_train_data, train_region_embed_id)
        plot_residuals(trn_resid, tst_resid, stdzd_test_data, stdzd_train_data)

    print("Test residuals mean {:5.4f}".format(np.mean(tst_resid)))
    print("Training residuals mean {:5.4f}".format(np.mean(trn_resid)))

    resid_neg = np.array(trn_resid)[np.array(stdzd_train_data).squeeze() < 0.0]
    resid_pos = np.array(trn_resid)[np.array(stdzd_train_data).squeeze() > 0.0]

    print("Training negative residuals mean {:5.4f}".format(np.mean(resid_neg)))
    print("Training positive residuals mean {:5.4f}".format(np.mean(resid_pos)))

    return mae, mse, round(np.mean(trn_resid), 5), round(np.mean(tst_resid), 5)


def standardize(x, allstats, vars):
    """
    Standardize numeric features based on ID.

    It is good practice to standardize features that use different scales and
    ranges. Although the model might converge without feature standardization,
    it makes training more difficult, and it makes the resulting model dependent
    on the choice of units used in the input.

    :param x:           A DataFrame with an 'ID' column
    :param allstats:    A DataFrame of summary statistics for each ID
    :param vars:        List of numeric variable names to standardize
    """
    assert isinstance(vars, list)

    for var in vars:
        stats = allstats[var]
        x = x.merge(stats[['mean', 'std']], left_on='ID', right_index=True)
        x.loc[:, var] = (x[var] - x['mean']) / x['std']
        x = x.drop(columns=['mean', 'std'])

    return x


def unstandardize(z, allstats, vars):
    """Inverse of standardize."""
    for var in vars:
        stats = allstats[var]
        z = z.merge(stats[['mean', 'std']], left_on='ID', right_index=True)
        z.loc[:, var] = z[var] * z['std'] + z['mean']
        z = z.drop(columns=['mean', 'std'])

    return z


def build_model(l1, l2, lr, embed_in_dim, embed_out_dim):
    """
    We need two separate inputs, one for categorical data that gets an
    embedding, and one for the numerical data.
    """
    input_num = layers.Input(shape=(1,))
    input_cat = layers.Input(shape=(1,))
    embed_layer = layers.Embedding(embed_in_dim, embed_out_dim)(input_cat)
    embed_layer = layers.Flatten()(embed_layer)
    merged_layer = layers.Concatenate()([input_num, embed_layer])
    output = layers.Dense(l1, activation=tf.keras.activations.relu)(merged_layer)
    output = layers.Dense(l2, activation=tf.keras.activations.relu)(output)
    output = layers.Dense(1, bias_initializer=tf.keras.initializers.constant(4.0))(output)

    model = keras.models.Model(inputs=[input_num, input_cat], outputs=[output])

    optimizer = tf.keras.optimizers.RMSprop(lr=lr)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def plot_residuals(trn_resid, tst_resid, stdzd_test_data, stdzd_train_data):
    """Plot residuals."""
    plt.xlabel('Standardized Temperature')
    plt.ylabel('Prediction residual')
    plt.scatter(stdzd_test_data, tst_resid, alpha=0.05)
    plt.scatter(stdzd_train_data, trn_resid, alpha=0.05)
    plt.savefig(f'{PLOTDIR}/residuals.png')
    plt.clf()


def plot_predictions_facet(model, category_names, allstats, stdzd_train_data, train_region_embed_id):
    """Facet plot of all temperature predictions."""
    temperature_seq = np.arange(-20, 40, 1) + 273.15
    npoints = len(temperature_seq)

    nrow = int(len(category_names)**0.5)
    ncol = int(len(category_names) / nrow + .999)

    plt.rcParams["figure.figsize"] = [20, 12]
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol)
    fig.suptitle('Modeled Load Predictions by Aggregate Region', fontsize=16)
    for i, region in enumerate(category_names):
        temp_data = pd.DataFrame({'ID': region, 'temp': temperature_seq})
        stdzd_seq_data = standardize(temp_data, allstats, ['temp']).temp.values
        stdzd_seq_predictions = model.predict([stdzd_seq_data, np.zeros(npoints) + i]).flatten()

        training_bounds = (min(stdzd_train_data.loc[train_region_embed_id == i, 'temp']),
                           max(stdzd_train_data.loc[train_region_embed_id == i, 'temp']))

        if len(category_names) == 1:
            ax = axes
        else:
            ax = axes[divmod(i, ncol)]
        ax.set_title(region)
        ax.axvspan(training_bounds[0], training_bounds[1], alpha=0.3, color='salmon')
        ax.set_xlim([-4, 4])
        ax.plot(stdzd_seq_data, stdzd_seq_predictions)

    fig.text(0.5, 0.04, 'Standardized Temperature', ha='center')
    fig.text(0.04, 0.5, 'Standardized Load', va='center', rotation='vertical')
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
    plt.ylabel('Mean Abs Error [Load (MW)]')
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

    args = get_args()

    diag_dir = resource_filename('eiafcst', os.path.join('models', 'diagnostic'))
    res_fname = os.path.join(diag_dir, 'gas_results.csv')

    # Record results always
    if not os.path.exists(res_fname):
        with open(res_fname, 'w') as results_file:
            results_file.write(','.join(['samplesize', 'lr', 'L1', 'L2', 'patience',
                                         'embedsize', 'mae', 'mse', 'mean train residual', 'mean test residual']))
            results_file.write('\n')

    with open(res_fname, 'a') as outfile:
        results = run(args.trainx, args.trainy, args.lr, args.L1, args.L2, args.epochs, args.patience,
                      args.model, args.embedsize, sample_size=sample_size, plots=True)

        argline = ','.join(str(a) for a in [sample_size, args.lr, args.L1, args.L2, args.patience, args.embedsize])
        outfile.write(argline + ',' + ','.join([str(r) for r in results]))
        outfile.write('\n')

        print(f'Done in {time.time() - st} seconds.')


if __name__ == '__main__':
    main()
