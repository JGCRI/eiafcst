#!/usr/bin/env python
#
# Query ECMWF for reanalysis temperature data for the United States.
#
# Bounding boxes for three separate regions are defined in bb.py. To run a
# request, provide three parameters:
#   1. region       one of 'hi', 'ak', or 'usa'
#   2. start_year   a year no earlier than 1979
#   3. end_year     a year no later than 2018
# The year restrictions are in place to prevent attempts at running invalid
# queries, however more years may become available at a later date, as ECMWF
# updates their database.
#
# To run, you will need a CDS API key as described here:
#   https://cds.climate.copernicus.eu/api-how-to
#
# Caleb Braun <caleb.braun@pnnl.gov>
# 1/28/19
import time
import sys
import cdsapi
from bb import CONTINENTAL_BB, HI_BB, AK_BB


# ====================================================================
# Read and validate user arguments
# ====================================================================

# Read arguments from command line.
try:
    region = sys.argv[1]
    start_year = int(sys.argv[2])
    end_year = int(sys.argv[3])
except IndexError:
    print("Must provide [hi|ak|usa] [start_year] [end_year]")
    sys.exit()

# Check year bounds
if start_year < 1979:
    print("Invalid start year")
    sys.exit()
if end_year > 2018:
    print("Invalid end year")
    sys.exit()
if end_year - start_year < 0:
    print("Invalid year range")
    sys.exit()

# bb.py contains bounding boxes in the form [x1, y1, x2, y2]
#                                         = [West, South, East, North]
if region == 'hi':
    area_bounds = HI_BB
elif region == 'ak':
    area_bounds = AK_BB
elif region == 'usa':
    area_bounds = CONTINENTAL_BB
else:
    print("Invalid region")
    exit()


# ====================================================================
# Prepare query variables
# ====================================================================

# Fix bounds order for API request, which expects [North, West, South, East]
area_bounds.insert(0, area_bounds.pop())

# Documentation on available variables can be found here:
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
var = "2m_temperature"

# Build year range from input arguments
years = [str(x) for x in range(start_year, end_year + 1)]

# Get all months, days, and hours (zero-padding is required)
months = [str(x + 1).zfill(2) for x in range(12)]     # Must be 01-12
days = [str(x + 1).zfill(2) for x in range(31)]       # Must be 01-31
times = [str(x).zfill(2) + ":00" for x in range(24)]  # Must be 00:00-23:00


# ====================================================================
# Request only two years at a time (because of request data limits)
# ====================================================================

year_iterator = iter(years)
for year in year_iterator:
    year_group = [year, next(year_iterator)]

    # Record how long it takes
    st = time.time()

    print(f"Requesting reanalysis {var} data with area bounds {area_bounds}")
    print(f"for years {year_group}")

    c = cdsapi.Client()

    c.retrieve('reanalysis-era5-single-levels', {
            "variable"     : var,
            "product_type" : "reanalysis",
            'year'         : year_group,
            'month'        : months,
            'day'          : days,
            "area"         : area_bounds,  # expects [North, West, South, East]. Default: global
            'time'         : times,
            "format"       : "netcdf"
        }, f"raw_data/{var}_{region}_{year_group[0]}-{year_group[-1]}.nc")

    elapsed = time.time() - st
    m, s = divmod(round(elapsed), 60)
    h, m = divmod(round(m), 60)
    total_time = ':'.join([str(x).zfill(2) for x in [h, m, s]])

    print(f"Finished in {total_time}")
