import pandas
import requests
import json
import numpy
from pandas.io.json import json_normalize
import re

print(pandas.__version__)
print(requests.__version__)
print(json.__version__)
print(numpy.__version__)

import pandas as pd 
import requests
import json
from pandas.io.json import json_normalize
import config
import numpy as np

class RidbData():

   def __init__(self, name, endpoint, url_params):
      self.df = pd.DataFrame()
      self.endpoint = endpoint
      self.url_params = url_params
      self.name = name

   def clean(self) :
      # by replacing '' with np.NaN we can use dropna to remove rows missing required data, like lat/longs
      self.df = self.df.replace('', np.nan)
    
      # normalize column names for lat and long. i.e. can be FacilityLatitude or RecAreaLatitude
      self.df.columns = self.df.columns.str.replace('.*Latitude', 'Latitude')
      self.df.columns = self.df.columns.str.replace('.*Longitude', 'Longitude')
      self.df = self.df.dropna(subset=['Latitude','Longitude'])

   def extract(self):
      request_url = self.endpoint
      response = requests.get(url=self.endpoint,params=self.url_params)
      data = json.loads(response.text)
      self.df = json_normalize(data['RECDATA'])

ridb_facilities_endpoint = 'https://ridb.recreation.gov/api/v1/facilities'
ridb_params = dict(apiKey= config.API_KEY)
ridb = RidbData('ridb', ridb_facilities_endpoint, ridb_params)

ridb.extract()

ridb.df.head()

ridb.df.shape

ridb.clean()

ridb.df.head()

ridb.df.shape

def get_ridb_data(endpoint,url_params):
   response = requests.get(url = endpoint, params = url_params)
   data = json.loads(response.text)
   df = json_normalize(data['RECDATA'])
   df = df.replace('', np.nan)
   df.columns = df.columns.str.replace('.*Latitude', 'Latitude')
   df.columns = df.columns.str.replace('.*Longitude', 'Longitude')
   df = df.dropna(subset=['Latitude','Longitude'])

   return df

ridb_df = get_ridb_data(ridb_facilities_endpoint, ridb_params)

ridb_df.head()

def get_ridb_facility_media(endpoint, url_params):
     # endpoint = https://ridb.recreation.gov/api/v1/facilities/facilityID/media/  
     response = requests.get(url = endpoint, params = url_params) 
     data = json.loads(response.text)
     df = json_normalize(data['RECDATA'])
     df = df[df['MediaType'] == 'Image']
     return df

ridb_media_endpoint = 'https://ridb.recreation.gov/api/v1/facilities/200006/media/'

ridb_df_media = get_ridb_facility_media(ridb_media_endpoint, ridb_params)

ridb_df_media

class RidbMediaData(RidbData):

   def clean(self) :
      self.df = self.df[self.df['MediaType'] == 'Image']

class RidbMediaData(RidbData):

    def clean(self) :
        self.df = self.df[self.df['MediaType'] == 'Image']

    def extract(self):
        request_url = self.endpoint
        for index, param_set in self.url_params.iterrows():
            facility_id = param_set['facilityID']
            req_url = self.endpoint + str(facility_id) + "/media"

            response = requests.get(url=req_url,params=dict(apiKey=param_set['apiKey']))
            data = json.loads(response.text)

            # append new records to self.df if any exist
            if data['RECDATA']:
                new_entry = json_normalize(data['RECDATA'])
                self.df = self.df.append(new_entry)

media_url = 'https://ridb.recreation.gov/api/v1/facilities/'
media_params = pd.DataFrame({
    'apiKey':config.API_KEY,
    'facilityID':[200001, 200002, 200003, 200004, 200005, 200006, 200007, 200008]
    })

ridb_media = RidbMediaData('media', media_url, media_params)

ridb_media.extract()

ridb_media.df

ridb_media.clean()

ridb_media.df

facilities_endpoint = 'https://ridb.recreation.gov/api/v1/facilities/'
recareas_endpoint = 'https://ridb.recreation.gov/api/v1/recareas'
key_dict = dict(apiKey = config.API_KEY)
facilities = RidbData('facilities', facilities_endpoint, key_dict)
recareas = RidbData('recareas', recareas_endpoint, key_dict)
facility_media = RidbMediaData('facilitymedia', facilities_endpoint, media_params) 

ridb_data = [facilities,recareas,facility_media]

# clean and extract all the RIDB data
list(map(lambda x: x.extract(), ridb_data))
list(map(lambda x: x.clean(), ridb_data))

facilities.df.head()

recareas.df.head()

facility_media.df.head()



