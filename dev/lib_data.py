import os
import julian
import datetime
import math
import pandas as pd
import numpy as np
from numpy import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import scipy.stats
from scipy.linalg import qr, solve, lstsq
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata
import random as rd
import julian
import time
from pylab import *
import warnings
import random
nasa_julian = 98
cnes_julian = 90
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
os.environ['PROJ_LIB']= "C:\\Users\\vankh\\Anaconda3\\Lib\\site-packages\\mpl_toolkits\\basemap"
from mpl_toolkits.basemap import Basemap
warnings.filterwarnings('ignore')

def region_ex(name,pres_max,fe = 0.1):
    """
    This function extracts data of a specific region !

    Parameters
    ----------
    - name: string
      name of the region
    - fe: float (default fe = 0.1)
      percentage of extracted dataset
    - pres_max: float
      maximum considered pressure

    Returns
    -------
    - region: dict
      contain all informations of the region
    """

    region_ = {}

    file = 'FetchData.pkl'

    if name == 'GulfStream':
      coords = {}
      coords["up_lon"] = -35
      coords["low_lon"] = -75
      coords["up_lat"] = 50
      coords["low_lat"] = 30
      name = 'GulfStream.pkl'
      region_ = regionv2(file,coords,name,fe,pres_max)
    elif name == 'Kuroshio':
      coords = {}
      coords["up_lon"] = 189
      coords["low_lon"] = 132
      coords["up_lat"] = 45
      coords["low_lat"] = 25
      name = 'Kuroshio.pkl'
      region_ = regionv2(file,coords,name,fe,pres_max)

    return region_


def regionv2( file, coords, name, fe, pres_max):
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

    - fe: float
      percentage of extracted dataset

    - pres_max: float
      maximum considered pressure

    Returns:
    --------
    - region: dict
      contain all region informations

    """

    # Get a particular region #
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

    # process days
    jul_days = region['juld']
    alpha0_x = np.ones(jul_days.shape[0])
    alpha1_x = jul_days
    w = 1/365.25
    sin_days = np.sin(2*math.pi*w*jul_days)
    cos_days = np.cos(2*math.pi*w*jul_days)

    # process latitude
    lat_rad = region['lat']*(math.pi/180)
    lat_sin = np.sin(lat_rad)

    # process longtitude
    lon_rad = region['lon']*(math.pi/180)
    lon_sin = np.sin(lon_rad)
    lon_cos = np.cos(lon_rad)

    # Create a processed dataset #
    region['alpha0_x'] = alpha0_x; region['alpha1_x'] = alpha1_x; region['sin_days'] = sin_days; region['cos_days'] = cos_days
    region['lat_sin'] = lat_sin; region['lon_sin'] = lon_sin; region['lon_cos'] = lon_cos
    data_split(region, pres_max, fe)


def data_split(region, pres_max, fe = 0.1):
  """
  Prepare the data to feed the model

  Parameters:
  ----------
  - region: dict
  - fe: float (default = 0.1)
      percentage of data to feed the model
  - pres_max: float
      maximum considered pressure

  Returns:
  --------
  - X: samples
  - y: target

  """
  # features = ['alpha0_x','alpha1_x','sin_days','cos_days','lat_sin','lon_sin','lon_cos','sla','beta0_p','beta1_p','beta2_p']
  features = ['alpha0_x','alpha1_x','sin_days','cos_days','lat_sin','lon_sin','lon_cos','sla','pres']
  targets = ['temp','psal']
  # orginal features
  o_features = ['lat','lon','juld','profile_ids']

  uniq_profile, _ = np.unique(region['profile_ids'], return_counts = True)
  X_train = np.empty(shape=[0, len(features)])
  y_train = np.empty(shape=[0, len(targets)])
  # X_test = np.empty(shape=[0, len(features)])
  # y_test = np.empty(shape=[0, len(targets)])
  o_feature_train = np.empty(shape=[0, len(o_features)])
  # o_feature_test = np.empty(shape=[0, len(o_features)])
  # profile pressure less than 100

  # Extract by profile_ids, and pressure less than 100
  for i,uni in enumerate(uniq_profile):

      index = np.where((region['profile_ids'] == uni)&(region['pres'] < pres_max))[0]
      nb_eles = int(index.shape[0]*fe)
      np.random.shuffle(index)

      train = index[0:nb_eles]
      # test = index[nb_eles:]

      x_uni_train = np.squeeze(np.asarray([[region[x][train]] for x in features])).T
      y_uni_train = np.squeeze(np.asarray([[region[x][train]] for x in targets])).T

      # x_uni_test = np.squeeze(np.asarray([[region[x][test]] for x in features])).T
      # y_uni_test = np.squeeze(np.asarray([[region[x][test]] for x in targets])).T

      o_fea_uni_train = np.squeeze(np.asarray([[region[x][train]] for x in o_features])).T
      # o_fea_uni_test = np.squeeze(np.asarray([[region[x][test]] for x in o_features])).T

      X_train = np.concatenate((X_train,x_uni_train.reshape(train.shape[0],len(features))), axis =0)
      y_train = np.concatenate((y_train,y_uni_train.reshape(train.shape[0],len(targets))), axis =0)

      # X_test = np.concatenate((X_test,x_uni_test.reshape(test.shape[0],len(features))), axis =0)
      # y_test = np.concatenate((y_test,y_uni_test.reshape(test.shape[0],len(targets))), axis =0)

      o_feature_train = np.concatenate((o_feature_train,o_fea_uni_train.reshape(train.shape[0],len(o_features))), axis = 0)
      # o_feature_test = np.concatenate((o_feature_test,o_fea_uni_test.reshape(test.shape[0],len(o_features))), axis = 0)

  sav_obj = open("x.pkl", 'wb')
  pickle.dump(X_train,sav_obj)
  sav_obj.close()

  sav_obj = open("y.pkl", 'wb')
  pickle.dump(y_train,sav_obj)
  sav_obj.close()

  sav_obj = open("feature.pkl", 'wb')
  pickle.dump(o_feature_train,sav_obj)
  sav_obj.close()


def lon_lat_juld(name):
    """
    """
    coords = {}
    file = 'FetchData.pkl'

    if name == 'GulfStream':
        coords = {}
        coords["up_lon"] = -35
        coords["low_lon"] = -75
        coords["up_lat"] = 50
        coords["low_lat"] = 30
        name = 'GulfStream_Coords.pkl'
        # Get a particular region #
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

        coords['lon'] = region['lon']
        coords['lat'] = region['lat']
        coords['juld'] = region['juld']
        coords['profile_ids'] = region['profile_ids']

    elif name == 'Kuroshio':
        coords = {}
        coords["up_lon"] = 189
        coords["low_lon"] = 132
        coords["up_lat"] = 45
        coords["low_lat"] = 25
        name = 'Kuroshio_Coords.pkl'
        # Get a particular region #
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

        coords['lon'] = region['lon']
        coords['lat'] = region['lat']
        coords['juld'] = region['juld']
        coords['profile_ids'] = region['profile_ids']

    # save dataset as a .pkl extention
    sav_obj = open(name, 'wb')
    pickle.dump(coords,sav_obj)
    sav_obj.close()

    return region


def DR(X, variance = 0.8, nb_max = 6, to_plot = False,):
  """
  This function does the dimension reduction on the samples

  Parameters
  ----------
  - X: numpy array
      (nb_samples,features)
  - variances: float (default = 0.8)
      The percentage of variances to keep
  - nb_max: int (default = 5)
      max number of components considered to plot
  - to_plot: boolean
      plot the analysis

  Returns
  -------
  - X_new: the new X with reduced dimensions
  """
  # number of observations
  n = X.shape[0]

  # instanciation
  acp = PCA(svd_solver='full')
  X_transform = acp.fit_transform(X)
  print("Number of acp components features= ", acp.n_components_)
  #variance explained
  eigval = acp.explained_variance_

  # variance of each component
  variances = acp.explained_variance_ratio_

  # percentage of variance explained
  cumsum_var_explained= np.cumsum(variances)
  print("cumsum variance explained= ",cumsum_var_explained[0:nb_max-1])

  #get the number of components satisfying the establised variance condition
  nb_component_features = np.where(cumsum_var_explained>variance)[0][0]
  acp_features = PCA(svd_solver='full',n_components =nb_component_features+1)
  X_new = acp_features.fit_transform(X)

  if to_plot:

      plt.figure(figsize=(10,5))
      plt.plot(np.arange(1,nb_max),variances[0:nb_max-1])
      plt.scatter(np.arange(1,nb_max),variances[0:nb_max-1])
      plt.title("Variance explained by each component")
      plt.ylabel("Variance values")
      plt.xlabel("Component")

      #scree plot
      plt.figure(figsize=(15,10))

      plt.subplot(221)
      plt.plot(np.arange(1,nb_max),eigval[0:nb_max-1])
      plt.scatter(np.arange(1,nb_max),eigval[0:nb_max-1])
      plt.title("Scree plot")
      plt.ylabel("Eigen values")
      plt.xlabel("Factor number")

      plt.subplot(222)
      plt.plot(np.arange(1,nb_max),cumsum_var_explained[0:nb_max-1])
      plt.scatter(np.arange(1,nb_max),cumsum_var_explained[0:nb_max-1])
      plt.title("Total Variance explained")
      plt.ylabel("Variance values")
      plt.xlabel("Factor number")

  return X_new


def spatial_dist(ft,Nlat,Nlon,regname, minlat, maxlat, minlon, maxlon):
    """
    Plot spatial distribution on the maps

    ft: feature train
    regnam: region name
    """

    # extract latitude and longtitude
    lon = ft[:,1]
    lat = ft[:,0]

    # create a grid of coordinates approximating the true coordinates
    xlat = np.linspace(min(lat),max(lat),Nlat)
    xlon = np.linspace(min(lon),max(lon),Nlon)
    # number of profiles at each approximated coordinate
    Freq = np.zeros((xlat.shape[0],xlon.shape[0]))


    N = lon.shape[0]

    for k in range(N):
        i = np.argmin(abs(xlat - lat[k]))
        j = np.argmin(abs(xlon - lon[k]))
        Freq[i,j] += 1

    # and plot !
    figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    map = Basemap(projection='merc', llcrnrlat=minlat, urcrnrlat=maxlat, llcrnrlon=minlon, urcrnrlon=maxlon, resolution='c')
    ax = plt.gca()
    plon, plat = map(xlon, xlat)
    xxlon,xxlat = meshgrid(plon,plat)
    map.scatter(xxlon, xxlat, c = Freq, marker = 's', cmap = "Blues")
    map.drawcoastlines()
    plt.title("Number of profiles in "+regname)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    plt.show()



def temporal_dist(ft, regname):

    """
    Plot temporal distribution on the maps

    ft: feature train
    regname: region name
    """

    # encoded julian days
    encoded_juld = [julian.from_jd(round(x), fmt='mjd') for x in ft[:,2]]

    months = np.asarray([x.month for x in encoded_juld])
    years = np.asarray([x.year + cnes_julian for x in encoded_juld])

    years_ = np.linspace(min(years),max(years),max(years) - min(years) + 1, dtype = int)
    months_ = np.linspace(1,12,12, dtype = int)

    count_months = np.zeros(months_.shape[0], dtype = int)
    count_years = np.zeros(years_.shape[0]*months_.shape[0], dtype = int)

    for m in list(months_):
        count = size(np.where(months == m))
        count_months[m-1] = count

    x_labels = ['Jan','Feb','Mar','Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.figure(figsize=(9, 6))
    plt.bar(x_labels,count_months);
    plt.title("Monthly Data Distribution in "+regname);
    plt.ylabel("Nb of profiles");

    i = 0
    for y in list(years_):
        for m in list(months_):
            count = size(np.where((years == y)*(months == m)))
            count_years[i] = count
            i = i + 1

    u_year = np.unique(years_)
    spots = np.asarray([12*i for i in range(len(u_year))])

    plt.figure(figsize=(12, 6))
    plt.plot(count_years);
    plt.grid()
    plt.xticks(spots,u_year);
    plt.title('Monthly Data Distribution in '+regname);
    plt.ylabel('Nb of profiles');



def train_test_split(X,y,features,test_per):

    """
    Split data into train and test sets
    """

    mask = [i for i in range(len(X))]
    random.shuffle(mask)

    mask_test = mask[:int(len(X)*test_per)]
    mask_train = mask[int(len(X)*test_per):]

    x_train, x_test = X[mask_train,:], X[mask_test,:]
    y_train, y_test = y[mask_train,:], y[mask_test,:]
    feature_train, feature_test = features[mask_train,:], features[mask_test,:]

    sav_obj = open("x_train.pkl", 'wb')
    pickle.dump(x_train,sav_obj)
    sav_obj.close()

    sav_obj = open("x_test.pkl", 'wb')
    pickle.dump(x_test,sav_obj)
    sav_obj.close()

    sav_obj = open("y_train.pkl", 'wb')
    pickle.dump(y_train,sav_obj)
    sav_obj.close()

    sav_obj = open("y_test.pkl", 'wb')
    pickle.dump(y_test,sav_obj)
    sav_obj.close()

    sav_obj = open("feature_train.pkl", 'wb')
    pickle.dump(feature_train,sav_obj)
    sav_obj.close()

    sav_obj = open("feature_test.pkl", 'wb')
    pickle.dump(feature_test,sav_obj)
    sav_obj.close()
