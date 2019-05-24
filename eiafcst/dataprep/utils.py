"""
Utility functions for data manipulation.

Caleb Braun
4/22/19
"""
import pandas as pd
import numpy as np
from os import path
from pkg_resources import resource_filename

from tensorflow.keras.callbacks import Callback


class PrintDot(Callback):
    """Display training progress."""

    def on_epoch_end(self, epoch, logs):
        """Print a single dot for each completed epoch."""
        if epoch % 100 == 0:
            print()
        print('.', end='', flush=True)


def add_quarter_and_week(df, datecol):
    """
    Add quarter and week column to DataFrame.

    1. Figure out the middle day of the week of each data point
    2. Use that day to assign the quarter of each data point
    3. Use that day to assign the week of the year of each data point
    4. Make the first week of each quarter restart at 1
    """
    cnames = list(df.columns)

    df['delta'] = pd.to_timedelta(3 - df[datecol].dt.weekday, unit='days')
    df['midweek'] = df[datecol] + df['delta']
    df['quarter'] = df['midweek'].dt.quarter
    df['week'] = df['midweek'].dt.week
    df['EconYear'] = df['midweek'].dt.year

    df.loc[:, 'week'] = df.groupby(['EconYear', 'quarter'])['week'].transform(lambda x: x - min(x) + 1)

    return df[['EconYear', 'quarter', 'week'] + cnames]


def fill_simulated(df, new_df):
    """
    Replace bad data with simulated values.

    The PCAs OVEC and WACM contain values that would cause the model to pick up
    unrealistic patterns in the data. For example, some years of WACM data are
    missing or all zero. The OVEC values remained constant across a year.
    Whether or not these represent the truth, we replace them with simulated
    values (from a random forest model) to keep totals and a realistic pattern.

    :param df:      Original DataFrame with bad data
    :new_df:        DataFrame of containing values to replace
    """
    df = pd.read_pickle('load_by_sub_region_2006-2017.pkl')
    new_df = pd.read_csv('random-forest-fill-values.csv')

    new_df['datetime'] = pd.to_datetime(new_df['datetime'], utc=True)

    new_df_col_map = {
        'year': 'EconYear',
        'quarter': 'quarter',
        'week': 'week',
        'nercrgn': 'NERC Region',
        'pca': 'Abbreviated PCA Name',
        'datetime': 'Hourly Load Data As Of',
        'load': 'Load (MW)'
    }

    # Ensure columns have common names, so that replacement maps correctly
    new_df = new_df.rename(columns=new_df_col_map)
    new_df = new_df[list(new_df_col_map.values())]

    # Set common columns as index, except for Load, which we are replacing
    idx_cols = list(new_df.columns)
    idx_cols.remove('Load (MW)')

    # Set identifier columns as index for the update function
    df_idx = df.set_index(idx_cols)
    new_df_idx = new_df.set_index(idx_cols)

    # Perform the update, and reset index columns as attribute columns
    df_idx.update(new_df_idx)
    df_fixed = df_idx.reset_index()

    return df_fixed[df.columns]


def remove_incomplete_weeks(df, datecol, aggcols):
    HR_IN_WK = 168
    return df.groupby(aggcols + ['EconYear', 'quarter', 'week']).filter(lambda x: len(x) == HR_IN_WK)


def long_to_wide(df, data_col='year'):
    """Convert EIA electricity data from long form to wide form."""
    key_cols = ['NERC Region', 'Master BA Name', 'Abbreviated PCA Name', 'eia_code']
    df = df.pivot_table(index=key_cols, columns=data_col, values='Load (MW)')
    df.reset_index(inplace=True)

    return df


def agg_to_year(df, key_cols=['NERC Region', 'Master BA Name', 'Abbreviated PCA Name', 'eia_code', 'year']):
    """
    Aggregate to year.

    Expects a clean DataFrame with the columns defined by key_cols.
    """
    df['year'] = df['Hourly Load Data As Of'].dt.year
    df = df.groupby(key_cols, as_index=False).agg({'Load (MW)': 'sum'})

    return df


def read_training_data(f, prec='float32'):
    """
    Read in a training dataset for modeling.

    Generally, we don't need float64 precision, so we can reduce it for efficiency.

    :param f:      path to .pkl or .csv file containing tidy dataset
    :param prec:   precision for numeric variables
    """
    if f.endswith('.csv'):
        dataset = pd.read_csv(f)
    else:
        dataset = pd.read_pickle(f)

    # Convert float columns to specified precision
    float_cols = (dataset.dtypes == np.float)
    dataset.loc[:, float_cols] = dataset.loc[:, float_cols].astype(prec)

    # Convert date/time column to datetime
    time_col = dataset.columns.str.lower().isin(['date', 'time'])

    if any(time_col):
        time_col = list(dataset.columns[time_col])[0]
        dataset.loc[:, time_col] = pd.to_datetime(dataset[time_col], utc=True, infer_datetime_format=True)

    return dataset


class DiagnosticFile:
    """
    A file for model diagnostics.

    This class provides an interface to a .csv file that records the diagnostic
    results of training a model with varying parameters.

    The file has three main groups of columns:
      1. The set of hyperparameters, including the paths to the training data.
      2. The metrics returned by the model training
      3. A column for any additional notes
    """

    def __init__(self, fname, hyperparams, res_metrics):
        """
        Construct a .csv file for recording diagnostic results of a model training.

        Checks if a diagnostic file exists, and creates one if not.

        :param fname:          the name of the diagnostic file.
        :param hyperparams:    list of hyperparameters that will be recorded
        :param res_metrics:    list of result metrics that will be recorded
        """
        diag_dir = resource_filename('eiafcst', path.join('models', 'diagnostic'))
        self.fname = path.join(diag_dir, fname)
        self.columns = hyperparams + res_metrics + ['notes']

        try:
            results_file = pd.read_csv(self.fname)
            if len(results_file.columns) != len(self.columns) or not all(results_file.columns == self.columns):
                raise ValueError(f'{fname} has columns\n{list(results_file.columns)}\nnot\n{self.columns}')
        except FileNotFoundError:
            results_file = pd.DataFrame(columns=self.columns)
            results_file.to_csv(self.fname, index=False)

    def write(self, hyper_values, results, notes):
        """
        Write the results of a model training.

        :param hyper_values:    the training run's hyperparameter values
        :param results:         the training run's diagnostic metrics
        :param notes:           any notes to be added to final column
        """
        diag_values = hyper_values + list(results) + [notes]

        results_file = pd.read_csv(self.fname)
        results_file.loc[results_file.index.max() + 1] = diag_values
        results_file.to_csv(self.fname, index=False, float_format='%.4f')


def combine_elec_and_temperature(elec, temp):
    """
    Prepare the dataset for modelling electric load from temperature.

    Prerequisits are data created from
        python eiafcst/main.py - e - fillfile eiafcst/data/random-forest-fill-values.csv ./
    and
        python eiafcst/dataprep/temperature.py temperature_by_agg_region.csv eiafcst/data/spatial/Aggregate_Regions/Aggregate_Regions.shp
    """
    temp = pd.read_csv(resource_filename('eiafcst', '../temperature_by_agg_region.csv'))
    load = pd.read_csv(resource_filename('eiafcst', '../load_by_agg_region_2006-2017.csv'))

    assert all(load['NERC Region'].isin(temp.ID))

    ONEHOT = False

    # Convert both to datetime
    load['time'] = pd.to_datetime(load['Hourly Load Data As Of'])
    temp['time'] = pd.to_datetime(temp['time'])

    # Rename and select for merging
    load = load.rename(columns={'NERC Region': 'ID'})
    load = load[['ID', 'time', 'Load (MW)']]
    temp = temp[['ID', 'time', 'temperature']]

    # Inner join the two data set
    data = load.merge(temp)

    if ONEHOT:
        # Convert our categorical ID column to numeric one-hot encoding
        data = pd.concat([data, pd.get_dummies(data['ID'])], axis=1)
        data = data.drop(columns='ID')

    data.to_csv(resource_filename('eiafcst', '../load_and_temp_full.csv'), index=False)
    data.to_pickle(resource_filename('eiafcst', '../load_and_temp_full.pkl'))


def plot_history(history, xlab='Epoch', ylab='Mean Abs Error', cols=[], labs=[], savefile=None, ylim=1.0):
    """
    Visualize the model's training progress using the stats stored in the
    history object.
    """
    import matplotlib.pyplot as plt

    if len(labs) != len(cols):
        labs = cols

    hist = pd.DataFrame(history.history)

    plt.figure()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.ylim((0, ylim))
    plt.title('Training Performance')

    for i, var in enumerate(cols):
        plt.plot(hist.index, hist[var], label=labs[i])

    plt.legend()

    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()

    plt.clf()
