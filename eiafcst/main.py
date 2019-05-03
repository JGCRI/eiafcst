"""
Run the GDP prediction model.

Caleb Braun
3/19/19
"""
from eiafcst.dataprep import electricity
from eiafcst.dataprep import natural_gas
from eiafcst.dataprep.utils import add_quarter_and_week
from eiafcst.models import model_gas

import pandas as pd

from pkg_resources import resource_filename
import argparse
import os


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", type=str, help="Directory for data outputs")
    parser.add_argument("-g", action="store_true", help="Run gas model")
    parser.add_argument("-e", action="store_true", help="Build electricity dataset")
    parser.add_argument("-fillfile", type=str, help="Replacement data for electricity dataset")
    # parser.add_argument('-L1', type=int,
    #                     help='The number of units in the first hidden layer (int) [default: 16]',
    #                     default=16)
    # parser.add_argument('-L2', type=int,
    #                     help='The number of units in the first hidden layer (int) [default: 16]',
    #                     default=16)
    # parser.add_argument("-epochs", type=int,
    #                     help="The number of epochs to train for", default=1000)
    # parser.add_argument('-patience', type=int,
    #                     help='How many epochs to continue training without improving dev accuracy (int) [default: 20]',
    #                     default=20)
    # parser.add_argument('-embedsize', type=int,
    #                     help='How many dimensions the region embedding should have (int) [default: 5]',
    #                     default=5)
    # parser.add_argument('-model', type=str,
    #                     help='Save the best model with this prefix (string) [default: /training_1/model.ckpt]',
    #                     default=os.path.normpath('/training_1/model.ckpt'))

    return parser.parse_args()


def elec(outdir, fill_file=None):
    """
    Produce electricity dataset, or replace with new values.

    :param outdir:      Where to find/save the load data
    :param fill_file:   If provided, used to replace subset of load values.
    """
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
            df = electricity.build_electricity_dataset()

    if fill_file is None and not build:
        print("Complete dataset already found.")
        return

    if fill_file is not None and not build:
        # Filling is based on datetime, not year/quarter/week designation
        df = df.drop(columns=['EconYear', 'quarter', 'week'])

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
    df_yearly_wide.to_csv(resource_filename('eiafcst', 'data/diagnostic/eia_yearly_wide.csv'), index=False)

    return df


def main():
    args = get_args()

    if not os.path.exists(args.outdir):
        raise NotADirectoryError

    if args.g:
        if not os.path.isfile('gas_hourly_by_state.csv'):
            print('Preparing natural gas dataset...')
            ngas = natural_gas.main()
            ngas.to_csv('gas_hourly_by_state.csv', index=False)

        model_gas.main()

    if args.e:
        print('Processing electricity dataset')
        elec_processed = elec(args.outdir, args.fillfile)


if __name__ == '__main__':
    main()
