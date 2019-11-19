import os
import julian
import datetime
import math
import pandas as pd
import numpy as np
import pickle


def region(file,coords,name):
    """
    Extract the the data of specific region from the global one by referencing to it's coordinates
    Preprocessing the data (date type, latiude, longtitude)
    
    Parameters:
    -----------
    - file: string
        a file name containing the global information of the ocean.
        
    - coords: dict 
        coordinates of the region.
        
    - name: string
        pickle file name of the region
        
    Returns:
    --------
    - region: dict 
        contain all region informations
    """
    #
    # get a particular region
    #
    # get the root path
    root = os.path.abspath(os.path.join("", '..'))
    # invoke the global data
    gl_data = pd.read_pickle(root+"\\data\\"+file)
    # delete rebundant informations
    del gl_data['n'],gl_data['profile_n']
    # get index
    index = np.where((gl_data['lat'] > coords["low_lat"])*(gl_data['lat'] < coords["up_lat"])*(gl_data['lon']>coords["low_lon"])*(gl_data['lon']<coords["up_lon"]))
    # region extract
    region = {v: gl_data[v][index] for v in gl_data}
     
    #
    # pre-process the region's data 
    #
    # date data pre-processing
    encoded_juld = [julian.from_jd(round(x), fmt='mjd') for x in region['juld']]
    days_in_year =  np.asarray([x.day + (x.month -1)*30 for x in encoded_juld])
    year = np.asarray([x.year for x in encoded_juld])
    # normalize the days and year
    norm_diy = days_in_year/365
    norm_y = (year - min(year))/(max(year)-min(year))    
    # process latitude
    lat_rad = region['lat']*(math.pi/180)
    lat_sin = np.sin(lat_rad)   
    # process longtitude
    lon_rad = region['lon']*(math.pi/180)
    lon_sin = np.sin(lon_rad)
    lon_cos = np.cos(lon_rad)
    # profile ids
    profile_ids = region['profile_ids']
    
    #
    # create a processed dataset
    #
    del region['lat'],region['lon'],region['juld'],region['profile_ids'],region['pres']
    region['norm_diy'] = norm_diy; region['norm_y'] = norm_y
    region['lat_sin'] = lat_sin; region['lon_sin'] = lon_sin; region['lon_cos'] = lon_cos
    region['profile_ids'] = profile_ids
    
    sav_obj = open(name, 'wb')
    pickle.dump(region,sav_obj)
    sav_obj.close()
    return region