"""
Run the GDP prediction model.

Caleb Braun
3/19/19
"""
from dataprep import electricity
from dataprep import natural_gas
from models import model_gas
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

    df.to_pickle(os.path.join(args.output_dir, 'load_by_sub_region_2006-2017.pkl'))
    # df = pd.read_csv('/Users/brau074/Documents/EIA/eiafcst/load_by_sub_region_2006-2017.csv')

    # Aggregate to NERC region
    keys = ['NERC Region', 'EconYear', 'quarter', 'week', 'Hourly Load Data As Of']
    dfagg = df.groupby(keys, as_index=False).agg({'Load (MW)': 'sum'})
    dfagg.to_pickle(os.path.join(args.output_dir, 'load_by_agg_region_2006-2017.pkl'))

    # Diagnostics
    df_yearly = electricity.agg_to_year(df)
    df_yearly_wide = electricity.long_to_wide(df_yearly)
    df_yearly_wide.to_csv(resource_filename('eiafcst', 'data/diagnostic/eia_yearly_wide.csv'), index=False)


if __name__ == '__main__':
    main()
