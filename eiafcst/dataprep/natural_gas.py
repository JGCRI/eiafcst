"""
Read the EIA natural gas data from spreadsheets into pandas objects.

Caleb Braun
1/7/19
"""
import os
import pandas as pd
from eiafcst.dataprep.utils import add_quarter_and_week, merge_temperature
from pkg_resources import resource_filename


def read_natural_gas_xls():
    """
    Collect EIA natural gas data from original excel spreadsheets.

    Gathers all data and outputs as tidy csv.
    """
    # Get path of excel input files
    gas_data_dir = resource_filename('eiafcst', 'data/raw_data/')
    gas_data_file = os.path.join(gas_data_dir, 'NG_CONS_SUM_A_EPG0_VCS_MMCF_M.xls')
    gas_data_sheet = "Data 1"

    # Read file, get recent years, and convert to long format
    df = pd.read_excel(gas_data_file, sheet_name=gas_data_sheet, skiprows=2)
    df = df[(df.Date >= '1998') & (df.Date < '2018')]
    df = pd.melt(df, id_vars='Date', var_name='State')

    assert all(df.State.str.contains('Natural Gas .* \(MMcf\)', regex=True))

    # The actual state should be everything except the words right before '(MMcf)'
    df.State = df.State.str.replace('.* in (.*) \(MMcf\)', '\\1', regex=True)
    df.State = df.State.str.replace('the ', '', regex=False)

    # Filter out USA total
    df = df[(df.State != 'U.S.')]

    return df.reset_index(drop=True)


def month_to_day(df):
    """Spread monthly values over days."""
    # Interestingly, note that adding pd.DateOffset(day=31) will not always
    # result in a date that ends on day 31. If the month is February, adding
    # pd.DateOffset(day=31) returns the last day in February.
    start_date = df.Date.min() - pd.DateOffset(day=1)
    end_date = df.Date.max() + pd.DateOffset(day=31)

    df = df.pivot(index='Date', columns='State', values='value')

    # reindex and fill the new days in with zeros
    dates = pd.date_range(start_date, end_date, freq='D')
    dates.name = 'Date'
    df = df.reindex(dates).fillna(0)

    df = df.reset_index().melt(id_vars='Date')

    # The maximum value of each month is the total from the original data
    # that we need to divide over all the days in the month
    df['month'] = df.Date.dt.month
    df['year'] = df.Date.dt.year
    df['value'] = df.groupby(['State', 'year', 'month'])['value'].transform(lambda x: max(x) / len(x))
    df = df.drop(columns=['month', 'year'])

    return df


def main():
    gasdf = read_natural_gas_xls()
    gasdf.Date = pd.to_datetime(gasdf.Date)
    gasdf = month_to_day(gasdf)
    gasdf = add_quarter_and_week(gasdf, datecol='Date')

    # Remove any incomplete weeks on the edges
    gasdf = gasdf.groupby(['EconYear', 'quarter', 'week', 'State']).filter(lambda x: len(x) == 7)

    # Now that we have complete economic weeks, aggregate to that level
    gasdf = gasdf.groupby(['EconYear', 'quarter', 'week', 'State'], as_index=False).agg({'value': 'sum'})

    return gasdf


if __name__ == '__main__':
    main()
