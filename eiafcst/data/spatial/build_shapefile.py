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


def nerc_name_to_abbr(df):
    """
    Convert the spatial data's name information from long name to abbreviation.

    The columns NAME and SUBNAME contain the abbreviation in parenthesis.
    """
    rgx = r"\(([A-Za-z]+)\)"
    df.loc[:, 'NAME_ABBR'] = df['NAME'].str.extract(rgx, expand=False)
    df[:, 'SUBNAME_ABBR'] = df['NAME_ABBR'] + "-" + df['SUBNAME'].str.extract(rgx, expand=False)

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


def main(root):
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
    bas_file = f'{root}data/electricity/EIA data work/EIA861/f8612017/Balancing_Authority_2017.xlsx'
    ba_to_nerc_file = f'{root}data/spatial/Balancing_Authority_to_NERC_Region.csv'

    print(f'\t{bas_file}')
    bas = pd.read_excel(bas_file)

    print(f'\t{ba_to_nerc_file}')
    ba_to_nerc = pd.read_csv(ba_to_nerc_file)

    # Spatial datasets
    ba_geo_file = f'{root}data/spatial/Control_Areas/Control_Areas.shp'
    nerc_geo_file = f'{root}data/spatial/NERC_Regions/NERC_Regions.shp'

    print(f'\t{ba_geo_file}')
    ba_geo = gpd.read_file(ba_geo_file)

    print(f'\t{nerc_geo_file}')
    nerc_geo = gpd.read_file(nerc_geo_file)

    # -------------------------------------------------------------------------
    print('Aggregating NYISO sub-regions...')
    nerc_geo = combine_nyiso(nerc_geo)
    print('\tDone.')

    # -------------------------------------------------------------------------
    print('Mapping Balancing Authorities to aggregate regions...')

    # Adjust metadata for mapping
    nerc_geo = nerc_name_to_abbr(nerc_geo)

    # NERC shapefile covers continental US, but we can grab AK and HI data from
    # the Balancing Authority shapefile
    ak_hi_geo = ba_geo.loc[ba_geo['STATE'].isin(['AK', 'HI']), :]
    ak_hi_geo.loc[:, 'ID'] = ak_hi_geo['ID'].astype('int')

    # The list of Balancing Authorities contains extra information for state
    # and year, but we just need the unique names and ids
    bas = bas[['BA ID', 'BA Code', 'Balancing Authority Name']].drop_duplicates()

    # Make meta information for Alaska and Hawaii BAs more standard
    ak_hi_geo = ak_hi_geo.merge(bas, left_on='ID', right_on='BA ID')
    ak_hi_geo = ak_hi_geo[['Balancing Authority Name', 'BA Code', 'BA ID', 'BA Code', 'geometry']]
    ak_hi_geo.columns = list(ba_to_nerc.columns) + ['geometry']

    # Add Alaska and Hawaii BAs as aggregate regions
    bas_to_agg = pd.concat([ba_to_nerc, ak_hi_geo.drop(columns='geometry')])

    # Filter to just the BAs
    bas_to_agg = bas_to_agg[bas_to_agg['EIA ID'].isin(bas['BA ID'])].reset_index(drop=True)

    print('\tDone.')

    # -------------------------------------------------------------------------
    print('Dissolving NERC subregions...')

    # The NERC shapefile contains NERC subregions, but we only use them if there
    # are Balancing Authorities that map to them (only WECC).
    use_subregion = nerc_geo['SUBNAME_ABBR'].isin(bas_to_agg['NERC Region'])
    nerc_geo.loc[use_subregion, 'NAME_ABBR'] = nerc_geo.loc[use_subregion, 'SUBNAME_ABBR']

    # Combine NERC subregions to aggregate NERC region (except WECC)
    nerc_agg_geo = nerc_geo[['NAME', 'NAME_ABBR', 'geometry']].dissolve(by=['NAME', 'NAME_ABBR'], as_index=False)

    # Add AK and HI aggregate regions to the NERC regions
    ak_hi_geo = ak_hi_geo[['Master BA Name', 'NERC Region', 'geometry']]
    ak_hi_geo = ak_hi_geo.rename(columns={'Master BA Name': 'NAME', 'NERC Region': 'NAME_ABBR'})

    assert nerc_agg_geo.crs == ak_hi_geo.crs
    final_gdf = gpd.GeoDataFrame(pd.concat([nerc_agg_geo, ak_hi_geo], ignore_index=True), crs=nerc_agg_geo.crs)

    # -------------------------------------------------------------------------
    print('Writing outputs...')

    plot_file = f'{root}data/spatial/agg_regions.png'
    shape_dir = f'{root}data/spatial/Aggregate_Regions'
    bas_to_agg_file = f'{root}data/spatial/ba_to_agg_region.csv'

    print(f'\t{plot_file}')
    plot_regions(final_gdf, plot_file)
    print(f'\t{shape_dir}')
    final_gdf.to_file(shape_dir, driver='ESRI Shapefile')
    print(f'\t{bas_to_agg_file}')
    bas_to_agg.to_csv(bas_to_agg_file, index=False)


if __name__ == '__main__':
    root = '/Users/brau074/Documents/EIA/'
    print(f'Running shapefile creation script in {root}')

    main(root)

    print('\n' + '=' * 10 + ' GOODBYE AND THANKS FOR ALL THE FISH ' + '=' * 10 + '\n')
