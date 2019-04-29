"""
Run the GDP prediction model.

Caleb Braun
3/19/19
"""
from eiafcst.dataprep import electricity
from eiafcst.dataprep import natural_gas
from eiafcst.dataprep.utils import add_quarter_and_week
from eiafcst.models import model_gas
from pkg_resources import resource_filename
import argparse
import os


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", type=str, help="Directory for data outputs")
    parser.add_argument("-g", action="save_true", help="Run gas model")
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
    return

    df = electricity.build_electricity_dataset()

    # Aggregate to NERC region
    keys = ['NERC Region', 'Hourly Load Data As Of']
    dfagg = df.groupby(keys, as_index=False).agg({'Load (MW)': 'sum'})
    dfagg = add_quarter_and_week(dfagg, datecol='Hourly Load Data As Of')
    dfagg = dfagg.groupby(['NERC Region', 'EconYear', 'quarter', 'week']).filter(lambda x: len(x) == 168)
    dfagg.to_pickle(os.path.join(args.outdir, 'load_by_agg_region_2006-2017.pkl'))
    dfagg.to_csv(os.path.join(args.outdir, 'load_by_agg_region_2006-2017.csv'), index=False)

    df = add_quarter_and_week(df, datecol='Hourly Load Data As Of')
    df = df.groupby(['eia_code', 'EconYear', 'quarter', 'week']).filter(lambda x: len(x) == 168)
    df.to_pickle(os.path.join(args.outdir, 'load_by_sub_region_2006-2017.pkl'))
    df.to_csv(os.path.join(args.outdir, 'load_by_sub_region_2006-2017.csv'), index=False)

    # Diagnostics
    df_yearly = electricity.agg_to_year(df)
    df_yearly_wide = electricity.long_to_wide(df_yearly)
    df_yearly_wide.to_csv(resource_filename('eiafcst', 'data/diagnostic/eia_yearly_wide.csv'), index=False)


if __name__ == '__main__':
    main()
