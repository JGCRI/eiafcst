"""
Prepare the electricity dataset.

Includes functions for converting the EIA electricity data from spreadsheets
into HDF5 format (requires PyTables >= 3.0.0.).

The output file from this document can easily be read from disk using:
>>> pd.read_hdf(hdf_name, hdf_tbl)

Caleb Braun
3/19/19
"""
import os
import time
import pandas as pd
import numpy as np
from pkg_resources import resource_filename


def xlsx_to_hd5(hdf_name, hdf_tbl):
    """
    Collect EIA electricity data from original excel spreadsheets.

    Combines all sheets into single dataframe and outputs in the HDF5 format.
    """
    # Get paths of all excel input files
    eia_data_dir = resource_filename('eiafcst', 'data/raw_data/')
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


def remove_pcas(df, pcas):
    """
    Remove given PCAs from the dataset.

    :param pcas:    List of Abbreviated PCA Name values to remove.
    """
    sub_region_rows = df['Abbreviated PCA Name'].isin(pcas)
    df = df.loc[~sub_region_rows, :].reset_index(drop=True)

    return df


def fix_southern_company(df):
    """
    Combine PCAs that all belong to Southern Company.

    Aggregate the sub-regions that are a part of it.
    """
    sub_region_codes = ['ALP', 'GAP', 'GPC', 'MSP', 'SP']
    sub_region_rows = df['Abbreviated PCA Name'].isin(sub_region_codes)
    df_sub = df.loc[sub_region_rows, :]
    df = df.loc[~sub_region_rows, :]

    key_cols = ['Year', 'Sequence Number', 'Hourly Load Data As Of']
    df_sub = df_sub.groupby(key_cols).agg({'Load (MW)': 'sum'})

    # Build to match the Southern Company rows in the large df
    df_sub.reset_index(inplace=True)
    df_sub['Power Control Area'] = 'Southern Company'
    df_sub['Abbreviated PCA Name'] = 'SOCO'

    key_cols += ['Power Control Area', 'Abbreviated PCA Name']

    # Creates 'Load (MW)_x' and 'Load (MW)_y'
    df = df.merge(df_sub, on=key_cols, how='outer')
    df['Load (MW)'] = df['Load (MW)_y'].fillna(df['Load (MW)_x'])

    df = df.drop(['Load (MW)_x', 'Load (MW)_y'], axis=1)
    assert all(~np.isnan(df['Load (MW)']))

    return df


def fix_wisconsin_miso(df):
    """Fix duplicate/inconsistant row in MISO - Wisconsin."""
    # Remove wrong values
    df = df.loc[~(df['Abbreviated PCA Name'] == 'WPS'), :]
    df['Power Control Area'].replace('MISO - Wisconsin Electric Power Company',
                                     'MISO - Wisconsin Public Service Corporation', inplace=True)
    df['Abbreviated PCA Name'].replace('WEC', 'WPS', inplace=True)

    return df


def fix_pacificorp(df):
    """
    Note that Pacificorp is already aggregated.

    The SNL data has PacifiCorp as one entity, when for most purposes
    PacifiCorp East (PACE) and PacifiCorp West (PACW) report as two
    entities. In this exercise, we consider it its own BA.
    """
    df['Power Control Area'] = df['Power Control Area'].replace('PacifiCorp E+W', 'PacifiCorp')
    return df


def fix_miso(df):
    """
    Aggregate MISO sub-entities into one group.

    MISO sub-entities are described in the file electricity/from_Aaron/BAs 2006-2014.xlsx
    """
    sub_region_codes = np.array(['MP', 'ALTE', 'ALTW', 'AMIL', 'AMMO', 'CWLD', 'CWLP', 'FE', 'GRE', 'HE', 'IPL', 'MGE',
                                 'NIPS', 'NSP', 'OTP', 'SIPC', 'SIGE', 'SMP', 'UPPC', 'WPS', 'MEC', 'MPW', 'DPC',
                                 'BREC', 'CNWY', 'LAFA', 'DENL', 'WMUC', 'CLEC', 'EDE', 'EES', 'LEPA', 'SME', 'SPPC'])
    sub_region_rows = df['Abbreviated PCA Name'].isin(sub_region_codes)
    df.loc[sub_region_rows, 'Power Control Area'] = 'Midcontinent Independent System Operator, Inc.'
    df.loc[sub_region_rows, 'Abbreviated PCA Name'] = 'MISO'

    key_cols = list(df.columns)
    key_cols.remove('Load (MW)')
    df = df.groupby(key_cols).agg({'Load (MW)': 'sum'})
    df.reset_index(inplace=True)

    return df


def add_missing(df, ferc_id_map, ferc_demand_file):
    """
    Add missing data found in the 714 database.

    Look for dates in the FERC-714 Part 3 Schedule 2 dataset that are not in
    the SNL data, and add them in.
    """
    id_map = pd.read_csv(ferc_id_map)
    ferc = pd.read_csv(ferc_demand_file)

    # All clues point to FERC having swapped the mapping for WAPA Desert
    # Southwest and WAPA UGP West so let's swap them back
    id_map.loc[id_map.respondent_id == 274, 'eia_code'] = 25471
    id_map.loc[id_map.respondent_id == 275, 'eia_code'] = 19610

    # Select only ID, date, and hourly data
    hours = ['hour' + str(h).zfill(2) for h in range(1, 26)]
    cols = ['respondent_id', 'plan_date', 'timezone'] + hours
    ferc = ferc.loc[:, cols]

    # We can't add missing years if we can't map to the respondent id; drop those rows
    df = df.dropna()
    df.loc[:, 'eia_code'] = df['eia_code'].astype('int')

    # Fix date column type
    df['Hourly Load Data As Of'] = pd.to_datetime(df['Hourly Load Data As Of'], format='%m/%d/%Y %I:%M:%S %p')
    ferc['plan_date'] = pd.to_datetime(ferc['plan_date'], format='%m/%d/%Y %H:%M:%S')

    # Map on column for respondent name and eia code; if we can't map it we can't use it
    ferc_mapped = ferc.merge(id_map, how='left')
    ferc_mapped = ferc_mapped.dropna()
    ferc_mapped['eia_code'] = ferc_mapped['eia_code'].astype('int')

    # Remove rows where the eia code is not in the SNL data
    ferc_mapped = ferc_mapped[ferc_mapped['eia_code'].isin(df['eia_code'])]

    # Gather hour columns so table is in long form
    ferc_long = ferc_mapped.melt(['respondent_name', 'respondent_id', 'eia_code', 'timezone', 'plan_date'],
                                 var_name='hour', value_name='Load (MW)')

    # NOTE: We are ignoring hour 25 because it is unclear what it is
    ferc_long = ferc_long[ferc_long['hour'] != 'hour25']
    ferc_long['hour'] = ferc_long['hour'].str[-2:].astype('int')

    # Add the hour to the date
    ferc_long['Hourly Load Data As Of'] = ferc_long['plan_date'] + pd.to_timedelta(ferc_long['hour'] - 1, unit='h')

    # Get the rows that are only in the FERC data, but need to be added to SNL
    ferc_only = pd.merge(ferc_long, df, how='left', indicator=True, on=['eia_code', 'Hourly Load Data As Of'])
    ferc_only = ferc_only.loc[ferc_only._merge == 'left_only', :]  # anti-join
    ferc_only = ferc_only.rename(columns={'Load (MW)_x': 'Load (MW)'})
    ferc_only = ferc_only[['eia_code', 'Hourly Load Data As Of', 'Load (MW)']]

    # For some reason, the Balancing Authorities Electric Energy Inc (EEI) and
    # WAPA Colorado-Missouri (WACM) have reported several years of data where
    # load values are all zero. For EEI, these are 2006, 2013-2017 and for
    # WACM they are 2008 (2007 is missing entirely). Regardless of whether this
    # is a reporting error or not, we can't explain it with the data we have,
    # and will negatively impact the performace of our model. Therefore we will
    # not add in these zeros, and they will remain as missing data to be dealt
    # with later.
    ferc_only = ferc_only[ferc_only['Load (MW)'] != 0]

    # Merge the correct ID columns back in
    snl_id_map = df.drop(columns=['Year', 'Sequence Number', 'Hourly Load Data As Of', 'Load (MW)'])
    snl_id_map = snl_id_map.drop_duplicates()
    ferc_only = ferc_only.merge(snl_id_map, on='eia_code')
    ferc_only['Year'] = ferc_only['Hourly Load Data As Of'].dt.year
    ferc_only['Sequence Number'] = 0  # This gets removed later
    ferc_only = ferc_only[df.columns]

    # Combine with original and return
    df = pd.concat([df, ferc_only])

    return df


def convert_to_utc(df, timezone_map):
    """
    Convert time stamps to UTC.

    Each PCA reports data in local time (likely), but see temp_elec_plots.py for further analysis.
    """
    # Shows that there are some PCAs with incomplete days (most in 2002 for daylight savings)
    if False:
        testdf = df.copy()
        testdf['day'] = testdf['Hourly Load Data As Of'].dt.day
        testdf['month'] = testdf['Hourly Load Data As Of'].dt.month
        bad = testdf.groupby(['eia_code', 'Year', 'month', 'day']).filter(lambda x: len(x) != 24)
        bad_summarized = bad.groupby(['eia_code', 'Year', 'month', 'day']).agg(['count'])
        bad_summarized.to_csv('pcas_missing_hours.csv')

    timezones = pd.read_csv(timezone_map)
    timezones = timezones.drop(columns='respondent_id')
    df = df.merge(timezones, how='left', on='eia_code')

    assert ~df.timezone.isnull().any()

    # Data appears to be in local standard time.
    load = 'Hourly Load Data As Of'
    df.loc[df.timezone == 'EST', load] = df.loc[df.timezone == 'EST', load] + pd.Timedelta('5 hours')
    df.loc[df.timezone == 'CST', load] = df.loc[df.timezone == 'CST', load] + pd.Timedelta('6 hours')
    df.loc[df.timezone == 'MST', load] = df.loc[df.timezone == 'MST', load] + pd.Timedelta('7 hours')
    df.loc[df.timezone == 'PST', load] = df.loc[df.timezone == 'PST', load] + pd.Timedelta('8 hours')
    df.loc[df.timezone == 'AKST', load] = df.loc[df.timezone == 'AKST', load] + pd.Timedelta('9 hours')
    df.loc[df.timezone == 'HST', load] = df.loc[df.timezone == 'HST', load] + pd.Timedelta('11 hours')
    df['Hourly Load Data As Of'] = df.loc[:, 'Hourly Load Data As Of'].dt.tz_localize('UTC')

    df = df.drop(columns=['eTag ID', 'timezone'])
    return df


def long_to_wide(df, data_col='year'):
    """Convert EIA electricity data from long form to wide form."""
    key_cols = ['NERC Region', 'Master BA Name', 'Abbreviated PCA Name', 'eia_code']
    df = df.pivot_table(index=key_cols, columns=data_col, values='Load (MW)')
    df.reset_index(inplace=True)

    return df


def agg_to_year(df):
    """
    Aggregate to year.

    Expects a clean DataFrame with the columns defined by key_cols.
    """
    df['year'] = df['Hourly Load Data As Of'].dt.year
    key_cols = ['NERC Region', 'Master BA Name', 'Abbreviated PCA Name', 'eia_code', 'year']
    df = df.groupby(key_cols, as_index=False).agg({'Load (MW)': 'sum'})

    return df


def map_eia_code(df, codemap):
    """
    Map on the column 'eia_code' to the SNL dataset.

    Note that this will add rows as some of the codes are double-mapped.
    """
    name_to_eia_code = pd.read_csv(codemap, names=['eia_code', 'Abbreviated PCA Name'])

    name_to_eia_code.pivot_table(['eia_code'], 'Abbreviated PCA Name')
    # Faster than merging
    mapping_dict = name_to_eia_code.pivot_table(['eia_code'], 'Abbreviated PCA Name', aggfunc='last')
    mapping_dict = mapping_dict.to_dict()['eia_code']
    df['eia_code'] = df['Abbreviated PCA Name']
    df['eia_code'] = df['eia_code'].map(mapping_dict)

    # Can't do anything with NaN eia_codes
    df = df.loc[~pd.isnull(df['eia_code']), :]
    df['eia_code'] = df['eia_code'].astype('int')
    df.reset_index(drop=True, inplace=True)

    return df


def filter_to_bas(df, ba_to_agg_map):
    """
    Map on the column NERC region and filter rows without a mapping.

    The "NERC region" is actually a combination of NERC regions, NERC
    subregions, and balancing authorities.
    """
    ba_to_agg = pd.read_csv(ba_to_agg_map)

    df = ba_to_agg.merge(df, left_on='EIA ID', right_on='eia_code')
    cols = ['NERC Region', 'Master BA Name'] + [c for c in df.columns if c not in ['NERC Region', 'Master BA Name']]
    df = df.loc[:, cols]
    df = df.drop(columns='Power Control Area')

    return df


def load_snl(hdf_name, hdf_tbl, min_year):
    """Load and prepare the SNL dataset for processing."""
    try:
        raw_elec = pd.read_hdf(hdf_name, hdf_tbl)
    except FileNotFoundError:
        xlsx_to_hd5(hdf_name, hdf_tbl)
        raw_elec = pd.read_hdf(hdf_name, hdf_tbl)

    snl = raw_elec.loc[raw_elec.Year >= min_year, :]
    snl = snl.reset_index(drop=True)  # Drop sheet name index

    # We have no idea what this ID is for, so let's remove it
    snl = snl.drop(columns='Power Control Area OID')

    # Remove whitespace and non-ascii character
    snl['Abbreviated PCA Name'] = snl['Abbreviated PCA Name'].str.strip()
    snl['Power Control Area'] = snl['Power Control Area'].str.replace('–', '-')

    return snl


def build_electricity_dataset(min_year):
    """
    Main function for creating the electricity dataset for modeling

    Draws on many sources to be described here eventually.

    The raw data(called `snl`, referring to the company that prepared it) has
    a few issues. Let's try to deal with some of them. Note that there are
    many more problems with the pre-2006 years of data that are not handled in
    this script.
    """
    maps = 'data/mapping_files'
    ferc = 'data/form714-database'
    diag = 'data/diagnostics'

    timezone_map = resource_filename('eiafcst', f'{maps}/timezones.csv')
    eia_code_map = resource_filename('eiafcst', f'{maps}/ba_code_to_eia_id.csv')
    agg_region_map = resource_filename('eiafcst', f'{maps}/ba_to_agg_region.csv')
    ferc_id_map = resource_filename('eiafcst', f'{ferc}/Respondent IDs.csv')
    ferc_demand_file = resource_filename('eiafcst', f'{ferc}/Part 3 Schedule 2 - Planning Area Hourly Demand.csv')

    hdf_name = resource_filename('eiafcst', 'data/raw_data/eia_elec_by_pca.h5')
    hdf_tbl = 'eia_data_all'

    snl = load_snl(hdf_name, hdf_tbl, min_year)

    # Remove Canadian provinces (Alberta, British Columbia, Mexico, Ontario, Manitoba, Saskatchewan)
    canadian_pcas = ['AESO', 'BCHA', 'CFE', 'IESO', 'MHEB', 'SPC']
    snl = remove_pcas(snl, canadian_pcas)

    # Merge or remove disaggregate regions from dataset
    snl = fix_southern_company(snl)
    snl = fix_wisconsin_miso(snl)
    snl = fix_miso(snl)
    snl = fix_pacificorp(snl)

    # Map on eia code
    snl = map_eia_code(snl, eia_code_map)

    # Filter to just the balancing authorities we will end up aggregating
    snl = filter_to_bas(snl, agg_region_map)

    # Add in missing data that occurs in FERC 714 dataset
    snl = add_missing(snl, ferc_id_map, ferc_demand_file)

    # The column "Sequence Number" is just the hour of the year; because we
    # have a column for date time, we can drop it.
    snl = snl.drop(columns='Sequence Number')

    # Reset index to be sequential after all the additions and drops
    snl = snl.reset_index(drop=True)

    snlbak = snl.copy()

    # Create the complete index with all date/BA combinations
    all_codes = snl.eia_code.unique()
    all_dates = np.sort(snl['Hourly Load Data As Of'].unique())
    all_index = pd.MultiIndex.from_product((all_codes, all_dates), names=['eia_code', 'Hourly Load Data As Of'])

    # Reindexing puts NaNs in for missing data
    snl = snl.set_index(['eia_code', 'Hourly Load Data As Of']).reindex(all_index).reset_index()

    # fill in identifying info for each BA (all cols except for load and date)
    meta_cols = [c for c in snl.columns if c not in {'eia_code', 'Hourly Load Data As Of', 'Load (MW)'}]
    snl.loc[:, meta_cols] = snl.groupby('eia_code')[meta_cols].fillna(method='ffill').fillna(method='bfill')

    snl = convert_to_utc(snl, timezone_map)

    # Now that the date column is finalized, we don't need a Year column
    snl = snl.drop(columns='Year')

    # Sort
    snl = snl.sort_values(['Master BA Name', 'Hourly Load Data As Of'])

    return snl[['NERC Region', 'Master BA Name', 'Abbreviated PCA Name',
                'eia_code', 'Hourly Load Data As Of', 'Load (MW)']]
