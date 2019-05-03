"""
build_shapefile.py

Construct shapefile for aggregating electricity data to NERC regions plus
two regions for AK & HI.

The list of Balancing Authorities comes from Form EIA-861, retrieved from here:
    https://www.eia.gov/electricity/data/eia861/

The shapefiles for NERC regions and Balancing Authorities was obtained from HIFLD:
    https://hifld-geoplatform.opendata.arcgis.com/datasets/nerc-regions
    https://hifld-geoplatform.opendata.arcgis.com/datasets/control-areas

Caleb Braun
3/18/19
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

import os
from pkg_resources import resource_filename


def nerc_name_to_abbr(df):
    """
    Convert the spatial data's name information from long name to abbreviation.

    The columns NAME and SUBNAME contain the abbreviation in parenthesis.
    """
    rgx = r"\(([A-Za-z]+)\)"
    df.loc[:, 'NAME_ABBR'] = df['NAME'].str.extract(rgx, expand=False)
    df.loc[:, 'SUBNAME_ABBR'] = df['NAME_ABBR'] + "-" + df['SUBNAME'].str.extract(rgx, expand=False)

    return df


def plot_regions(nerc_agg, outf):
    fill_colors = ["#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99",
                   "#4392bb", "#75c384", "#e4bc44", "#eea7cd", "#d9b8ff", "#99f6ff"]
    fig, ax = plt.subplots(1, figsize=(20, 10))
    for i, geo in nerc_agg.centroid.iteritems():
        color = fill_colors[i % len(fill_colors)]
        if i == 0:
            ax = nerc_agg.iloc[[0], :].plot(color=color, edgecolor='black', figsize=(20, 10))
        else:
            nerc_agg.iloc[[i], :].plot(ax=ax, color=color, edgecolor='black')
        ax.annotate(s=nerc_agg['NAME'][i], xy=[geo.x, geo.y], color='black', ha='center')

    plt.savefig(outf)
    plt.close(fig)


def combine_nyiso(nerc_geo):
    """
    Combine sub-NYISO regions into one.

    The NERC region shapefile provides boundaries for multiple NYISO regions:
     1. LONG ISLAND (NYLI)
     2. UPSTATE NEW YORK (NYUP)
     3. NYC - WESTCHESTER (NYCW)
    We only have load data for NYISO as a whole, so we don't want this level
    of detail.
    """
    combine = nerc_geo.SUBNAME.isin(['LONG ISLAND (NYLI)', 'UPSTATE NEW YORK (NYUP)', 'NYC - WESTCHESTER (NYCW)'])
    nerc_geo.loc[combine, 'SUBNAME'] = 'NEW YORK INDEPENDENT SYSTEM OPERATOR (NYIS)'
    nerc_geo = nerc_geo.dissolve(by='SUBNAME', as_index=False)

    return nerc_geo


def build_mapping_file(eia_ba_xlsx_file):
    """
    Build the file ba_to_agg_region.csv.

    This function builds the file that maps from Balancing Authority to the
    aggregate regions used in this modelling process. The mapping file is
    included in this package by default, but this function shows the steps
    used to create it.

    The input to this function is the path to the official EIA list of
    Balancing Authorities, as reported in form EIA-861 data files. For example,
    the 2017 file is called Balancing_Authority_2017.xlsx. This data was
    obtained here:
        https://www.eia.gov/electricity/data/eia861/
    """
    bas = pd.read_excel(eia_ba_xlsx_file)

    # TODO: Implement/copy from backup file

    # The list of Balancing Authorities contains extra information for state
    # and year, but we just need the unique names and ids
    bas = bas[['BA ID', 'BA Code', 'Balancing Authority Name']].drop_duplicates()

    bas_to_agg_file = resource_filename('eiafcst', os.path.join('data', 'mapping_files', 'ba_to_agg_region.csv'))
    bas_to_agg.to_csv(bas_to_agg_file, index=False)

    return bas_to_agg


def main(outdir):
    """
    Create shapefile of aggregated regions.

    1. Read in mapping and spatial datasets
    2. Map Balancing Authorities to their aggregate NERC region
    3. Combine shapes of NERC subregions that we don't have detail for
    4. Combine NERC shapefile and BA shapefile for AK and HI
    5. Plot and save final shapefile

    The final dataset has two attributes(in addition to geometry)
    """
    # -------------------------------------------------------------------------
    print('Reading files...')

    # Region mapping datasets
    ba_to_nerc_file = resource_filename('eiafcst', os.path.join('data', 'mapping_files', 'ba_to_agg_region.csv'))

    print(f'\t{ba_to_nerc_file}')
    ba_to_nerc = pd.read_csv(ba_to_nerc_file)

    # Spatial datasets
    ba_geo_file = resource_filename('eiafcst', os.path.join('data', 'spatial', 'Control_Areas', 'Control_Areas.shp'))
    nerc_geo_file = resource_filename('eiafcst', os.path.join('data', 'spatial', 'NERC_Regions', 'NERC_Regions.shp'))

    print(f'\t{ba_geo_file}')
    ba_geo = gpd.read_file(ba_geo_file)

    print(f'\t{nerc_geo_file}')
    nerc_geo = gpd.read_file(nerc_geo_file)

    # -------------------------------------------------------------------------
    print('Aggregating NYISO sub-regions...')
    nerc_geo = combine_nyiso(nerc_geo)
    print('\tDone.')

    # -------------------------------------------------------------------------
    print('Dissolving NERC subregions...')

    bas_to_agg = pd.read_csv(ba_to_nerc_file)

    nerc_geo = nerc_name_to_abbr(nerc_geo)

    # The NERC shapefile contains NERC subregions, but we only use them if there
    # are Balancing Authorities that map to them (WECC and NPCC).
    use_subregion = nerc_geo['SUBNAME_ABBR'].isin(bas_to_agg['NERC Region'])
    nerc_geo.loc[use_subregion, 'NAME_ABBR'] = nerc_geo.loc[use_subregion, 'SUBNAME_ABBR']

    # Combine NERC subregions to aggregate NERC region (except WECC, NPCC gets split into NY and NE)
    nerc_agg_geo = nerc_geo[['NAME', 'NAME_ABBR', 'geometry']].dissolve(by=['NAME', 'NAME_ABBR'], as_index=False)
    nerc_agg_geo.plot()

    # Add AK and HI aggregate regions to the NERC regions. The NERC shapefile
    # covers continental US, but we can grab AK and HI data from the Balancing
    # Authority shapefile.
    ak_hi_geo = ba_geo.loc[ba_geo['STATE'].isin(['AK', 'HI']), :]
    ak_hi_geo.loc[:, 'ID'] = ak_hi_geo.loc[:, 'ID'].astype('int')
    ak_hi_geo = ak_hi_geo.merge(bas_to_agg, left_on='ID', right_on='EIA ID')
    ak_hi_geo = ak_hi_geo[['NAME', 'NERC Region', 'geometry']]
    ak_hi_geo = ak_hi_geo.rename(columns={'NERC Region': 'NAME_ABBR'})

    assert nerc_agg_geo.crs == ak_hi_geo.crs
    final_gdf = gpd.GeoDataFrame(pd.concat([nerc_agg_geo, ak_hi_geo], ignore_index=True), crs=nerc_agg_geo.crs)
    final_gdf = final_gdf.rename(columns={'NAME_ABBR': 'ID'})

    # -------------------------------------------------------------------------
    print('Writing outputs...')

    plot_file = f'{outdir}/agg_regions.png'
    shape_dir = f'{outdir}/Aggregate_Regions'

    print(f'\t{plot_file}')
    plot_regions(final_gdf, plot_file)
    print(f'\t{shape_dir}')
    final_gdf.to_file(shape_dir, driver='ESRI Shapefile')


if __name__ == '__main__':
    outdir = resource_filename('eiafcst', os.path.join('data', 'spatial'))
    print(f'Running shapefile creation script outputting in {outdir}')

    main(outdir)

    print('\n' + '=' * 10 + ' GOODBYE AND THANKS FOR ALL THE FISH ' + '=' * 10 + '\n')
