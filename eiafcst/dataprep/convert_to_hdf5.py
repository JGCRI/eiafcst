"""
Convert the EIA electricity data from spreadsheets into HDF5 format.

Requires PyTables >= 3.0.0.

The output file from this document can easily be read from disk using:
>>> pd.read_hdf(hdf_name, hdf_tbl)

Caleb Braun
1/7/19
"""
import pandas as pd
import numpy as np
import os
import glob
import time


def xlsx_to_hd5(hdf_name, hdf_tbl):
    """
    Collect EIA electricity data from original excel spreadsheets.

    Combines all sheets into single dataframe and outputs in the HDF5 format.
    """
    # Get paths of all excel input files
    eia_data_dir = "/Users/brau074/Documents/EIA/data/electricity/raw_data"
    eia_data_glob = os.path.join(eia_data_dir, 'SQL*.xlsx')
    xlsx_files = sorted(glob.glob(eia_data_glob))

    # Define column data types
    data_types = {
        'Year': np.int32,
        'Sequence Number': np.int32,
        'Load (MW)': np.float64,
        'Hourly Load Data As Of': 'str',
        'Power Control Area OID': np.int32,
        'Power Control Area': 'str',
        'Abbreviated PCA Name': 'str'
    }

    # Write out as pandas file
    store = pd.HDFStore(hdf_name)

    # Read in the files
    eia_data_all_dfs = []
    for xlf in xlsx_files:
        st = time.time()
        print(f'Reading in {xlf}...')
        eia_df_sheets = pd.read_excel(xlf, sheet_name=None, dtype=data_types)
        m, s = divmod(time.time() - st, 60)
        print('Finished in {:02.0f}:{:02.0f} seconds.'.format(m, s))

        # Combine
        eia_df = pd.concat(eia_df_sheets)
        eia_data_all_dfs.append(eia_df)

    eia_data_all = pd.concat(eia_data_all_dfs)

    store[hdf_tbl] = eia_data_all
    store.close()


if __name__ == '__main__':
    hdf_name = 'eia_elec_by_pca.h5'
    hdf_tbl = 'eia_data_all'

    xlsx_to_hd5(hdf_name, hdf_tbl=hdf_tbl)
