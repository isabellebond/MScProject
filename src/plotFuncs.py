import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt 
import json
from urllib.request import urlopen
import plotly.express as px
from geojson_rewind import rewind

class plot():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def scatter_plot(self, X, y, title = 'Title', save = None):
        for item in X:
            plt.plot(item, y, data = self.dataframe)
        plt.legend()
        plt.title(title)
        plt.show()

        if type(save) == str:
            plt.savefig(save)

        return
        
        

