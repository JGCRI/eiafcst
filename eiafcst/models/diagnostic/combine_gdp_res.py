# Combine csv files that contain the same columns.
#
# All .csv files starting with the first argument will be combined.
#
# Caleb Braun
# 07/03/19
import pandas as pd
import os
import sys

try:
	instart = sys.argv[1]
	outfile = sys.argv[2]
except IndexError:
	print('combine_gdp_res.py [instart] [outfile]')
	raise

all_files = os.listdir()
csv_files = [f for f in all_files if f.endswith('.csv')]
gdp_files = [f for f in csv_files if f.startswith(instart)]

results = [pd.read_csv(f) for f in gdp_files]
results = pd.concat(results)

results.to_csv(outfile, index=False)
