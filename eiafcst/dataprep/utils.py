"""
Utility functions for data manipulation.

Caleb Braun
4/22/19
"""
import pandas as pd


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
