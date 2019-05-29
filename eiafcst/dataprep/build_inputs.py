"""
Construct inputs for the GDP model.

This script needs to be run once before the model training process.

Caleb Braun
5/29/19
"""

import os
import numpy as np
from pkg_resources import resource_filename

from eiafcst.dataprep import FuelParser
from eiafcst.dataprep.utils import read_training_data, add_quarter_and_week
from eiafcst.dataprep.economic import parse_gdp


def load_fuel(fp, fuel_type, syear, eyear):
    """
    Prepare fuel data as input to the GDP model.

    Filters out categories like residential consumption and electric power, as
    those categories are presumably just repeating information provided by the
    electricity inputs.

    Standardizes and adds 'case' column, representing the quarter number.

    :param fp:          FuelParser object
    :param fuel_type:   One of 'gas' or 'petrol'
    :param syear:       Earliest year to include
    :param eyear:       Last year to include
    """
    fuel = fp.parse(fuel_type)

    fuel = fuel[fuel['EconYear'].isin(range(syear, eyear + 1))]

    val_col = fuel.columns[-1]
    var_col = fuel.columns[-2]

    # Filter to best GDP categories for natural gas
    if fuel_type == 'gas':
        gdp_categories = ['Industrial Consumption', 'Lease and Plant Fuel Consumption', 'Pipeline & Distrubtion Use']
        fuel = fuel[fuel[var_col].isin(gdp_categories)]

    # Aggregate categories to weekly total
    fuel = fuel.groupby(['EconYear', 'quarter', 'week'], as_index=False).agg({val_col: 'sum'})

    # Standardize
    fuel[val_col] = (fuel[val_col] - fuel[val_col].mean()) / fuel[val_col].std()

    # Add case (quarter) number
    fuel['case'] = (fuel['EconYear'] - fuel['EconYear'].min()) * 4 + fuel['quarter'] - 1

    fuel_vals = fuel[val_col]
    fuel_vals.index = fuel['case']

    return fuel_vals


def elec_residuals_to_cases(elec_residuals):
    """
    Convert electricity residuals DataFrame to array of cases.

    The GDP model needs the electricity data separated by economic quarters,
    which can be of different size. Extract the values from the DataFrame,
    where weeks, hours, and regions are separate dimensions in a Numpy array.

    :param elec_residuals:  DataFrame of regional electricity residuals.
    """
    HR_WK = 24 * 7  # hours in a week

    # Our goal is an array of cases [year-quarters] with dimensions [weeks, hours, regions]
    elec_residuals = elec_residuals.sort_values(['EconYear', 'quarter', 'week', 'time', 'ID'])
    nreg = len(elec_residuals.ID.unique())

    elec_arr = elec_residuals.residuals.values

    qtr_bounds = np.where(elec_residuals['quarter'].diff())[0]
    qtr_bounds = qtr_bounds[1:]  # Only need inside bounds for the split
    elec_lst = np.array_split(elec_arr, qtr_bounds)
    elec_lst = np.array([a.reshape(-1, HR_WK, nreg) for a in elec_lst])

    # Array with dimensions [weeks, regions]
    return elec_lst


def check_output_dir(pth):
    """
    Build output folders, if they don't already exist.

    Creates sub-directories in pth:
        train/
        dev/
        test/

    These folders hold the corresponding datasets for the GDP model.
    """
    for subdir in ['train', 'dev', 'test']:
        subpth = os.path.join(pth, subdir)
        if subdir not in os.listdir(pth):
            os.mkdir(subpth)
            continue

        datafiles = ['elec.npy', 'gas.npy', 'petrol.npy', 'gdp.npy']
        overwrite = any([os.path.isfile(os.path.join(subpth, f)) for f in datafiles])

        if overwrite:
            stop = input(f'Files in {subpth} will be overwritten, continue? (y/n)')
            if not len(stop) == 1 and stop[0] == 'y':
                raise SystemExit()


def main():
    """
    Prepare inputs for the GDP model.

    The GDP model requires 4 inputs:
      1. Hourly electricity load residuals by aggregate region
      2. Weekly natural gas consumption
      3. Weekly petroleum consumption
      4. Quarterly GDP
    """
    # TODO: Move these constants to a global package configuration file
    MODEL_INPUT_DIR = resource_filename('eiafcst', os.path.join('models', 'gdp'))
    ELEC_RESID_PATH = resource_filename('eiafcst', os.path.join('models', 'electricity', 'elec_model5_residuals.csv'))
    GDP_LABELS_PATH = resource_filename('eiafcst', os.path.join('data', 'raw_data', 'NQGDP_2002-2018_NSA.csv'))

    RNG_SEED = 42
    DEV_FRAC = 1 / 5
    TEST_FRAC = 1 / 5

    check_output_dir(MODEL_INPUT_DIR)

    elec_residuals = read_training_data(ELEC_RESID_PATH)
    elec_residuals = add_quarter_and_week(elec_residuals, 'time')

    syear = elec_residuals['EconYear'].min()
    eyear = elec_residuals['EconYear'].max()

    elec_residuals = elec_residuals_to_cases(elec_residuals)

    gdp = parse_gdp(GDP_LABELS_PATH, syear, eyear)

    fp = FuelParser.FuelParser()
    gas = load_fuel(fp, 'gas', syear, eyear)
    petrol = load_fuel(fp, 'petrol', syear, eyear)

    # Sanity check that all inputs match shapes
    assert len(elec_residuals) == len(gdp)
    assert len(gas) == len(petrol) == sum([x.shape[0] for x in elec_residuals])

    # Randomly choose which cases are for training the model, which are for
    # model validation, and which are set aside for testing. The test set
    # and validation sets each contain 1/5 of all cases. Therefore the validation
    # set is 1/4 the size of the training set.
    ncases = len(gdp)
    np.random.seed(RNG_SEED)
    rand_idx = np.random.permutation(ncases)  # 27, 40, 26, 43
    splits = [int(ncases * (1 - TEST_FRAC - DEV_FRAC)), int(ncases * (1 - TEST_FRAC))]
    train_idx, dev_idx, test_idx = np.split(rand_idx, splits)

    # Write each subset of 4 inputs to train, dev, and test directories
    for type, idx in [('train', train_idx), ('dev', dev_idx), ('test', test_idx)]:
        outdir = os.path.join(MODEL_INPUT_DIR, type)

        gas_subset = np.array([gas[i].values for i in idx])
        petrol_subset = np.array([petrol[i].values for i in idx])

        np.save(os.path.join(outdir, 'elec.npy'), elec_residuals[idx])
        np.save(os.path.join(outdir, 'gas.npy'), gas_subset)
        np.save(os.path.join(outdir, 'petrol.npy'), petrol_subset)
        np.save(os.path.join(outdir, 'gdp.npy'), gdp.gdp.values[idx])
        np.save(os.path.join(outdir, 'time.npy'), idx)


if __name__ == '__main__':
    main()
