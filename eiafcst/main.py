"""
Run data preparation for the GDP prediction model.

Caleb Braun
3/19/19
"""
from eiafcst.data.spatial import build_shapefile
from eiafcst.dataprep import electricity
from eiafcst.dataprep.utils import add_quarter_and_week

import pandas as pd

from pkg_resources import resource_filename
import argparse
import os


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-outdir", type=str, help="Directory for data outputs", default="eiafcst/output")
    parser.add_argument("-e", action="store_true", help="Build electricity dataset")
    parser.add_argument("-f", action="store_true", help="Download input datasets, even if they already exist")
    parser.add_argument("-s", action="store_true", help="Build the electricity regions shapefile")
    parser.add_argument("-fillfile", type=str, help="Replacement data for electricity dataset")

    return parser.parse_args()


def elec(outdir, diagdir, fill_file=None):
    """
    Produce electricity dataset, or replace with new values.

    :param outdir:      Where to find/save the load data
    :param fill_file:   If provided, used to replace subset of load values.
    """
    print('Processing electricity dataset')

    sub_region_pkl = os.path.join(outdir, 'load_by_sub_region_2006-2017.pkl')
    sub_region_csv = os.path.join(outdir, 'load_by_sub_region_2006-2017.csv')
    agg_region_pkl = os.path.join(outdir, 'load_by_agg_region_2006-2017.pkl')
    agg_region_csv = os.path.join(outdir, 'load_by_agg_region_2006-2017.csv')

    # Attempt to read electricity file from disk, otherwise build it
    build = False
    try:
        df = pd.read_pickle(sub_region_pkl)
    except FileNotFoundError:
        try:
            df = pd.read_csv(sub_region_csv)
        except FileNotFoundError:
            build = True
            df = electricity.build_electricity_dataset(2006)

    if fill_file is None and not build:
        print("Complete dataset already found.")
        return

    if fill_file is not None and not build:
        # Filling is based on datetime, not year/quarter/week designation
        df = df.drop(columns=['EconYear', 'quarter', 'week'])

    if fill_file is not None:
        df = electricity.fill_simulated(df, fill_file)

    # Aggregate to NERC region
    keys = ['NERC Region', 'Hourly Load Data As Of']
    dfagg = df.groupby(keys, as_index=False).agg({'Load (MW)': 'sum'})

    # Remove incomplete weeks. This needs to be done separately after
    # aggregating because some aggregate areas have subregions that cross
    # timezones.
    dfagg = add_quarter_and_week(dfagg, datecol='Hourly Load Data As Of')
    dfagg = dfagg.groupby(['NERC Region', 'EconYear', 'quarter', 'week']).filter(lambda x: len(x) == 168)

    dfagg.to_pickle(agg_region_pkl)
    dfagg.to_csv(agg_region_csv, index=False)

    # Remove incomplete weeks at subregional level.
    df = add_quarter_and_week(df, datecol='Hourly Load Data As Of')
    df = df.groupby(['eia_code', 'EconYear', 'quarter', 'week']).filter(lambda x: len(x) == 168)

    df.to_pickle(sub_region_pkl)
    df.to_csv(sub_region_csv, index=False)

    # Diagnostics
    df_yearly = electricity.agg_to_year(df)
    df_yearly_wide = electricity.long_to_wide(df_yearly)
    df_yearly_file = os.path.join(diagdir, 'eia_yearly_wide.csv')
    df_yearly_wide.to_csv(df_yearly_file, index=False)

    return df


def download_inputs(force):
    """Check if required datasets exist."""
    form714 = os.listdir(resource_filename('eiafcst', os.path.join('data', 'form714-database')))
    rawdata = os.listdir(resource_filename('eiafcst', os.path.join('data', 'raw_data')))
    spatial = os.listdir(resource_filename('eiafcst', os.path.join('data', 'spatial')))

    # Sets of required files
    required_form714 = set(['Part 3 Schedule 2 - Planning Area Hourly Demand.csv',
                            'Respondent IDs.csv'])
    required_rawdata = set(['SQL 1 1998-2002.xlsx',
                            'SQL 2 2003-2007.xlsx',
                            'SQL 3 2008-2012.xlsx',
                            'SQL 4 2013-2017.xlsx',
                            'NG_CONS_SUM_DCU_NUS_M.xls',
                            'NQGDP_2002-2018_NSA.csv',
                            'PET_CONS_WPSUP_K_W.xls'])
    required_spatial = set(['Control_Areas', 'NERC_Regions', 'states_21basic'])

    if required_spatial != required_spatial & set(spatial):
        raise FileNotFoundError('Required spatial datasets are missing.')
    elif required_rawdata != required_rawdata & set(rawdata):
        raise FileNotFoundError('Required raw datasets are missing.')
    elif required_form714 != required_form714 & set(form714):
        raise FileNotFoundError('FERC 714 data are missing.')

    return True


def main():
    """
    Generate all the data necessary to run the GDP model, then run it.

    Run this function to automatically generate several data files.
    Arguments are flags on how much of the system to run.
    """
    args = get_args()

    spatial_dir = resource_filename('eiafcst', os.path.join('data', 'spatial'))
    output_dir = args.outdir
    diag_dir = resource_filename('eiafcst', os.path.join('data', 'diagnostic'))

    for d in [spatial_dir, output_dir, diag_dir]:
        if not os.path.exists(d):
            raise NotADirectoryError(d)

    # Download all required datasets. Ideally, we can automate this (with
    # permission), but for now we just check if they exist.
    download_inputs(args.f)

    # Build the aggregate region shapefile
    if args.s:
        build_shapefile.main(spatial_dir)

    # Process electricity data to correct years and aggregation level
    if args.e:
        elec_processed = elec(output_dir, diag_dir, args.fillfile)


if __name__ == '__main__':
    main()
