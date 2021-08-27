import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import geopy
from geopy.geocoders import Nominatim
from kmodes.kmodes import KModes
import os
import pgeocode

class geo_plots():

    def __init__(self, geofile, OAfile):
        self.geofile = gpd.read_file(geofile)
        self.OApostcode = pd.read_csv(OAfile)[['Postcode', 'OA11CD']]
        self.features = {}
        self.target = {}
        self.data = {}
    
    def load_data(self, datafile):
        filename = os.path.splitext(os.path.basename(datafile))[0]
        self.data[filename] = pd.read_csv(datafile)
        try:
            self.target[filename] = self.data[filename]['Target']
            self.features[filename] = self.data[filename].drop('Target', axis = 1)
        except KeyError:
            pass
            
        return

    def plot_target(self, key):
        plot = self.data[key].groupby('OA11CD').mean()
        plot = plot.merge(self.geofile, on = ['OA11CD'], how = 'outer')
        plot.plot(column = 'Target')

    def cluster_data(self, num_clusters = [2,3,4,5,6,7,8,9,10]):
        for num in num_clusters:
            kmode = KModes(num_clusters=num)
            kmode.fit_predict(self.data)
            



    def lats_to_postcode(self):
        geo = pgeocode.Nominatim('gb')
        coordinates = self.data[['latitude','logitude']]
        postcode = geo.reverse(coordinates)
        self.data[postcode] = postcode
        return

    def convert_to_latlong(self, columname):
        geo = pgeocode.Nominatim('gb')
        self.data[columname] = self.dataframe[columname].apply(lambda x:' '.join([x[0:len(x)-3],x[len(x)-3:]]))
        #print(self.dataframe[columname])
        self.data['latitude'] = geo.query_postal_code(self.dataframe[columname].values)['latitude']
        self.data['longitude'] = geo.query_postal_code(self.dataframe[columname].values)['longitude']
        return


