"""
Utility functions for data manipulation.

Caleb Braun
4/22/19
"""
import pandas as pd
import numpy as np
from os import path
from pkg_resources import resource_filename


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
    df['week'] = df.groupby(['EconYear', 'quarter'])['week'].transform(lambda x: x - min(x) + 1)

    return df[['EconYear', 'quarter', 'week'] + cnames]


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


def diagnostic_file(fname, columns):
    """
    Construct a .csv file for recording diagnostic results of a model training.

    Checks if a diagnostic file exists, and creates one if not.

    :param fname:       the name of the diagnostic file.
    :param columns:     list of attributes that will be recorded
    """
    diag_dir = resource_filename('eiafcst', path.join('models', 'diagnostic'))
    res_fname = path.join(diag_dir, fname)

    columns[-1] += '\n'

    if not path.exists(res_fname):
        with open(res_fname, 'w') as results_file:
            results_file.write(','.join(columns))
    else:
        with open(res_fname, 'r') as results_file:
            header = results_file.readline().split(',')
            assert columns == header

    return res_fname
