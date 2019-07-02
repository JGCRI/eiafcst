"""
Produce weekly national temperature averages weighted by population.

The directory containing input temperatures must follow the naming
convention:
    2m_temperature_[ak|hi|usa]_[year]-[year + 1].nc

Caleb Braun
4/15/19
"""
from pkg_resources import resource_filename
import argparse
import os
from rasterio import features
from affine import Affine
import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np


# ==============================================================================
# Shapefile masking functions
#
# The following 3 functions are used to apply a shapefile-defined region
# coordinate to a lat/lon xarray DataArray.
#
# Source:
#   https://stackoverflow.com/questions/51398563/python-mask-netcdf-data-using-shapefile
# ==============================================================================
def transform_from_latlon(lat, lon):
    """Input 1D array of lat / lon and output an Affine transformation."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, latitude='latitude', longitude='longitude',
              fill=np.nan, **kwargs):
    """
    Rasterize a list of (geometry, fill_value) tuples onto the given
    xarray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir+shp_file)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : **kwargs (dict): passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.


    """
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}

    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))


def add_shape_coord_from_data_array(xr_da, shp_path, coord_name):
    """
    Create a new coord for the xr_da indicating whether or not it
    is inside the shapefile.

    Creates a new coord - "coord_name" which will have integer values
    used to subset xr_da for plotting / analysis/

    Usage:
    -----
    precip_da = add_shape_coord_from_data_array(precip_da, "awash.shp", "awash")
    awash_da = precip_da.where(precip_da.awash==0, other=np.nan)
    """
    # 1. read in shapefile
    shp_gpd = gpd.read_file(shp_path)

    # 2. create a list of tuples (shapely.geometry, id)
    #    this allows for many different polygons within a .shp file (e.g. States of US)
    shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]

    # shapes = list(shp_gpd[['geometry', 'NAME_ABBR']].apply(tuple, axis=1))
    # 3. create a new coord in the xr_da which will be set to the id in `shapes`
    xr_da[coord_name] = rasterize(shapes, xr_da.coords,
                                  longitude='longitude', latitude='latitude')

    id_col = guess_id_column(shp_gpd)
    return xr_da, {idx: name for idx, name in enumerate(shp_gpd.iloc[:, id_col])}


def guess_id_column(df):
    """Guess the index of the most useful identifying column of a DataFrame."""
    cols = list(df.columns.str.lower())
    try:
        return cols.index('id')
    except ValueError:
        try:
            return cols.index('name')
        except ValueError:
            return 0


def read_global_pop(ncfile):
    """
    Read global population from a NetCDF file.

    Load Gridded Population of the World (GPW) data in as a masked 2D numpy
    array. Extracts the 2015 global population grid as well as associated
    latitude and longitude values.

    Data source:
        Center for International Earth Science Information Network - CIESIN -
        Columbia University. 2018. Gridded Population of the World, Version 4
        (GPWv4): Population Count, Revision 11. Palisades, NY: NASA
        Socioeconomic Data and Applications Center (SEDAC).
        https://doi.org/10.7927/H4JW8BX5. Accessed 15 April 2019.

    :param ncfile:     Path to NetCDF file.
    :return:           DataArray of global population data
    """
    with xr.open_dataset(ncfile) as pop:
        # Convert population to DataArray
        pop = pop.to_array().squeeze()

        # The original NetCDF contains many rasters; the 4th one is 2015 population
        pop = pop.sel(raster=4).drop('raster')

        # NOTE: We add an eighth degree offset to align to the grids in the
        # temperature datasets.
        pop['latitude'] += 0.125
        pop['longitude'] += 0.125

    return pop


def average_da(self, dim=None, weights=None):
    """
    Weighted average for DataArrays.

    Source:
        https://github.com/pydata/xarray/issues/422#issuecomment-140823232

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of self.

    Returns
    -------
    reduced : DataArray
        New DataArray with average applied to its data and the indicated
        dimension(s) removed.

    """
    if weights is None:
        return self.mean(dim)
    else:
        if not isinstance(weights, xr.DataArray):
            raise ValueError("weights must be a DataArray")

        # if NaNs are present, we need individual weights
        if self.notnull().any():
            total_weights = weights.where(self.notnull()).sum(dim=dim)
        else:
            total_weights = weights.sum(dim)

        return (self * weights).sum(dim) / total_weights


def population_weighted_average(xr_da, pop=None):
    """
    Take weighted average using matching population grid cells.

    Selects the cells from the population array that match the cells of the
    given DataArray.

    :param xr_da:   An xarray DataArray, indexed by stacked latitude and
                    longitude
    :param pop:     An xarray DataArray of population values, indexed by
                    stacked latitude and longitude
    """
    if pop is not None:
        pop_region = pop.sel(stacked_latitude_longitude=xr_da.stacked_latitude_longitude)
        assert all(pop_region.stacked_latitude_longitude == xr_da.stacked_latitude_longitude)
        avg = average_da(xr_da, dim='stacked_latitude_longitude', weights=pop_region)
    else:
        avg = xr_da.mean(dim='stacked_latitude_longitude')

    return avg


def aggregate_temperature(indir, years, region, shp, pop=None):
    """
    Read gridded hourly temperature NetCDF for a single region.

    Given a region ('ak', 'hi', or 'usa'), extract average hourly temperature.
    Reads ALL NetCDF files matching '2m_temperature_region_*', where 'region'
    is one of the accepted US regions.

    Subregions, such as electricity control areas or states, should be provided
    as a shapefile.

    :param indir:   directory containing the temperature NetCDF files
    :param region:  one of 'ak', 'hi', or 'usa'
    :param shp:     path to shapefile containing subregions
    :param pop:     path to shapefile containing subregions
    """
    file_root = os.path.join(indir, f'2m_temperature_{region}_')
    nc_files = [f'{file_root}{y}-{y + 1}.nc' for y in range(years[0], years[1] + 1, 2)]

    # This module makes use of the package xarray for its particularly useful
    # features of reading multiple NetCDF files and maintaining labelled
    # dimensions. The files are read in as a Dataset object by default,
    # however the temperature data is more useful as a 3D DataArray.
    with xr.open_mfdataset(nc_files) as ds:
        temperature = ds.to_array().squeeze()

    # See hack note below
    dc = temperature[:, temperature.latitude == 39, temperature.longitude == -77].values

    # Add the shapes in shp as a new element of the DataArray's coords, named
    # 'region'. This is a non-dimension coordinate, unlike time, lat, and lon.
    temperature, regmap = add_shape_coord_from_data_array(temperature, shp, 'region')

    if pop is None:
        weights = None
    else:
        # When taking the temperature average by group, the latitude and
        # longitude dimensions get stacked by default. To use the population
        # data as weights, we need to stack the spatial dimensions similarly.
        weights = pop.stack(stacked_latitude_longitude=['latitude', 'longitude'])

    temperature = temperature.groupby('region').apply(population_weighted_average, args=[weights])

    # The work done above represents regions with integer encodings. Map back
    # to their actual string IDs using the region map returned above.
    tmp_df = temperature.to_dataframe('temperature').reset_index()
    tmp_df['ID'] = tmp_df.region.map(regmap)
    tmp_df = tmp_df[['ID', 'time', 'temperature']]

    # HACK alert! A common aggregation region is by state. The District of
    # Columbia is smaller than the resolution of the temperature data, and does
    # not get assigned to any cell. Rather than generalize this case, we will
    # instead just take the values of the grid cell covering DC and tack them
    # on to the end of the data. The assumption here is that if VA and MD are
    # spatial regions, then DC should be one too.
    if any(tmp_df.ID == 'Virginia') and any(tmp_df.ID == 'Maryland'):
        dc_df = tmp_df[tmp_df.ID == 'Maryland'].copy()
        dc_df.ID = 'District of Columbia'
        dc_df.temperature = dc.flatten()
        tmp_df = tmp_df.append(dc_df, sort=True).reset_index(drop=True)

    return tmp_df


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('outfile', type=str, help='file name (.csv) for aggregate temperature outputs')
    parser.add_argument('shp', type=str, help='subregional shapefile')
    parser.add_argument('-w', help='weight by population', action='store_true')
    # parser.add_argument('-timestep', help='Number of hours in each time step (default 24)', default=24)
    parser.add_argument('-y1', type=int, help='start year (default 2006)', default=2006)
    parser.add_argument('-y2', type=int, help='start year (default 2017)', default=2017)

    return parser.parse_args()


def main():
    """
    Build temperature datasets at different spatial resolutions.

    State dataset comes from:
    https://www.arcgis.com/home/item.html?id=f7f805eb65eb4ab787a0a3e1116ca7e5#overview
    """
    args = get_args()

    indir = resource_filename('eiafcst', os.path.join('data', 'raw_data', 'temperature'))

    # Load population data as weights
    if args.w:
        global_pop_ncfile = os.path.join(indir, 'gpw_v4_population_count_rev11_15_min.nc')
        global_pop = read_global_pop(global_pop_ncfile)
    else:
        global_pop = None

    years = (args.y1, args.y2)

    usa_avg = aggregate_temperature(indir, years, 'usa', args.shp, pop=global_pop)
    hi_avg = aggregate_temperature(indir, years, 'hi', args.shp, pop=global_pop)
    ak_avg = aggregate_temperature(indir, years, 'ak', args.shp, pop=global_pop)

    out = pd.concat([usa_avg, hi_avg, ak_avg], sort=True)
    out = out.sort_values(['ID', 'time'])

    # Add UTC indicator
    out.loc[:, 'time'] = out['time'].dt.tz_localize('UTC')

    out_fname = args.outfile
    if not out_fname.endswith('.csv'):
        out_fname += '.csv'

    out.to_csv(out_fname, float_format='%3.4f', index=False)


if __name__ == '__main__':
    main()
