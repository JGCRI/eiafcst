"""
Read the EIA fuel data from spreadsheets into pandas objects.

Caleb Braun
5/15/19
"""
import os
import pandas as pd
from eiafcst.dataprep.utils import add_quarter_and_week
from pkg_resources import resource_filename


class FuelParser:
    """
    Read EIA fuel data for use in models.

    Fuel consumption data is publically available on the EIA's website. This
    class expects any fuel to be used to be downloaded and placed in the
    package's raw_data directory.

    Currently, support is provided for natural gas and petroleum data, found
    respectively at:
        https://www.eia.gov/dnav/ng/ng_cons_sum_dcu_nus_m.htm
        https://www.eia.gov/dnav/pet/pet_cons_wpsup_k_w.htm
    Clicking the 'Download Series History' link provides the datasets needed
    for the GDP model.
    """

    def __init__(self, start_year=1998, end_year=2018):
        """
        Initialize a FuelParser.

        :param start_year:  First year of data to parse
        :param end_year:    Last year of data to parse
        """
        self.data_dir = resource_filename('eiafcst', 'data/raw_data/')
        self.data_sheet = 'Data 1'

        self.date_col = 'Date'  # The name of the column that holds the date

        self.year1 = start_year
        self.year2 = end_year

    def read_fuel_xls(self, file_name, var_name, unit, prefix):
        """
        Collect EIA fuel data from original excel spreadsheet.

        Read file, get recent years, and convert to long format.
        """
        data_file = os.path.join(self.data_dir, file_name)

        df = pd.read_excel(data_file, sheet_name=self.data_sheet, skiprows=2)
        df = df[(df[self.date_col] >= str(self.year1)) & (df[self.date_col] < str(self.year2))]
        df = pd.melt(df, id_vars=self.date_col, var_name=var_name)

        assert all(df[var_name].str.contains(rf'{prefix} .* \({unit}\)', regex=True))

        # The actual product name is after prefix
        df[var_name] = df[var_name].str.replace(rf'.*{prefix} (.*) \({unit}\)', '\\1', regex=True)

        return df

    def month_to_day(self, df, var_name):
        """Spread monthly values over days."""
        # Interestingly, note that adding pd.DateOffset(day=31) will not always
        # result in a date that ends on day 31. If the month is February, adding
        # pd.DateOffset(day=31) returns the last day in February.
        start_date = df[self.date_col].min() - pd.DateOffset(day=1)
        end_date = df[self.date_col].max() + pd.DateOffset(day=31)

        df = df.pivot(index=self.date_col, columns=var_name, values='value')

        # reindex and fill the new days in with zeros
        dates = pd.date_range(start_date, end_date, freq='D')
        dates.name = self.date_col
        df = df.reindex(dates).fillna(0)

        df = df.reset_index().melt(id_vars=self.date_col)

        # The maximum value of each month is the total from the original data
        # that we need to divide over all the days in the month
        df['month'] = df[self.date_col].dt.month
        df['year'] = df[self.date_col].dt.year
        df['value'] = df.groupby([var_name, 'year', 'month'])['value'].transform(lambda x: max(x) / len(x))
        df = df.drop(columns=['month', 'year'])

        return df

    def parse(self, fuel):
        """
        Run the parser for a given fuel.

        :param fuel:    One of 'gas' or 'petrol'
        """
        if fuel not in ['gas', 'petrol']:
            raise ValueError("Fuel must be one of 'gas' or 'petrol'")

        # Define data variables based on fuel
        if fuel == 'gas':
            file_name = 'NG_CONS_SUM_DCU_NUS_M.xls'
            var_name = 'end_use'
            unit = 'MMcf'
            prefix = 'Natural Gas'

        if fuel == 'petrol':
            file_name = 'PET_CONS_WPSUP_K_W.xls'
            var_name = 'product'
            unit = 'Thousand Barrels per Day'
            prefix = 'Weekly U.S. Product Supplied of'

        fuel_df = self.read_fuel_xls(file_name, var_name, unit, prefix)

        fuel_df[self.date_col] = pd.to_datetime(fuel_df[self.date_col])
        fuel_df = self.month_to_day(fuel_df, var_name)
        fuel_df = add_quarter_and_week(fuel_df, datecol=self.date_col)

        # Remove any incomplete weeks on the edges
        fuel_df = fuel_df.groupby(['EconYear', 'quarter', 'week', var_name]).filter(lambda x: len(x) == 7)

        # Now that we have complete economic weeks, aggregate to that level
        fuel_df = fuel_df.groupby(['EconYear', 'quarter', 'week', var_name], as_index=False).agg({'value': 'sum'})

        return fuel_df.rename(columns={'value': unit})
