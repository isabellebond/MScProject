import pandas as pd
import os
import numpy as np
import dataFuncs 
import json
from urllib.request import urlopen
import plotly.express as px
from geojson_rewind import rewind

print('done')
#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_rawdir = os.path.join(_datadir, 'rawdata')
print('done')
#Load GeoJson

N=100
with open(os.path.join(_datadir,'Output_Areas.geojson')) as f:
    for i in range(0, N):
        print(f.readline(i))
    OA = json.load(f)
    

#Load data to be charted 
income = pd.read_csv(os.path.join(_datadir,'Output_Areas.csv'))
ofcom = dataFuncs.csv_to_pd('graphtest',_preprocesseddir)
income['Mean Income'] = dataFuncs.['Mean Income']
print('done')
#Make the rings clockwwise (to make it compatible with plotly)    
OA_corrected=rewind(OA,rfc7946=False)
print('done')
fig = px.choropleth(income, geojson=OA_corrected, locations='OA11CD', featureidkey="properties.OA11CD", color='Mean Income',
                            color_continuous_scale="PurPor", labels={'label name':'label name'}, title='MAP TITLE',
                            scope="europe")
print('done')
fig.update_geos(fitbounds="locations", visible=False)
print('done')