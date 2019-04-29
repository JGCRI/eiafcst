Adding spatial dimensionality to our modelling inputs helps remove temperature
effects on energy use. For example, a hot day in Texas will result in an
increase in energy used for cooling, without necessarily (and likely not)
correlating with either productivity or energy use in Maine.

Our input datasets are at national, state, and control area detail. To extract
temperature information corresponding to these regions, we use a few different
shapefiles, described below:


Aggregate_Regions
The regions defined here are used for extracting temperature effects from the
electricity load data. The regions in this shapefile are a combination of
NERC regions, NERC subregions, and control areas. The script build_shapefile.py
is used to combine these regions.


Control_Areas
A shapefile of electric power control areas, also known as Balancing Authority
Areas. More information and the source of this data can be found here:
https://hifld-geoplatform.opendata.arcgis.com/datasets/control-areas


NERC_Regions
A shapefile representing North American Electric Reliability Corporation (NERC)
Regions and Subregions. More information and the source of this data can be
found here:
https://hifld-geoplatform.opendata.arcgis.com/datasets/nerc-regions


states_21basic
A shapefile of US 50 states and DC. More information and the source of this
data can be found here:
https://www.arcgis.com/home/item.html?id=f7f805eb65eb4ab787a0a3e1116ca7e5
