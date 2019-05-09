"""
Read the BEA GDP data from a csv file.

Caleb Braun
5/9/19
"""
import os
import pandas as pd
from pkg_resources import resource_filename


def parse_gdp(gdp_file=None, syear=2006, eyear=2017):
    """
    Collect non-seasonally adjusted GDP data.

    Source:
        Table 8.1.5. Gross Domestic Product, Not Seasonally Adjusted
        https://apps.bea.gov/iTable/index_nipa.cfm
    """
    if gdp_file is None:
        gdp_file = resource_filename('eiafcst', os.path.join('data', 'raw_data', 'NQGDP_2002-2018_NSA.csv'))

    skip_rows = 4

    # Read, clean labels, and make (year, quarter) the row index
    gdp = pd.read_csv(gdp_file, header=[0, 1], skiprows=skip_rows, index_col=1).drop(columns=1)
    gdp.index = gdp.index.str.strip()
    gdp = gdp.transpose()

    # Build labels for GDP values from (year, quarter) index
    gdp_labeled = pd.DataFrame(gdp.index.tolist(), columns=['EconYear', 'quarter'], index=gdp.index)
    gdp_labeled.loc[:, 'gdp'] = gdp['Gross domestic product']
    gdp_labeled = gdp_labeled.reset_index(drop=True)

    gdp_labeled.loc[:, 'EconYear'] = gdp_labeled['EconYear'].astype('int')
    gdp_labeled.loc[:, 'quarter'] = gdp_labeled['quarter'].str[1].astype('int')

    gdp_labeled = gdp_labeled[(syear <= gdp_labeled['EconYear']) & (gdp_labeled['EconYear'] <= eyear)]

    return gdp_labeled


if __name__ == '__main__':
    parse_gdp()
