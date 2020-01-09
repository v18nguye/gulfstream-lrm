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
import random as rd
import time
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PROJ_LIB']= "C:\\Users\\vankh\\Anaconda3\\Lib\\site-packages\\mpl_toolkits\\basemap"


def region_ex(name, fe = 0.1):
    """
    This function extracts data of a specific region !

    Parameters
    ----------
    - name: string
      name of the region
    - fe: float (default fe = 0.1)
      percentage of extracted dataset

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
      region_ = regionv2(file,coords,name,fe)
    elif name == 'Kuroshio':
      coords = {}
      coords["up_lon"] = 189
      coords["low_lon"] = 132
      coords["up_lat"] = 45
      coords["low_lat"] = 25
      name = 'Kuroshio.pkl'
      region_ = regionv2(file,coords,name,fe)

    return region_


def regionv2(file,coords,name,fe):
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

    # process pressure
    # pres = gl_data['pres']
    ## mode for 0 - 100
    # beta0_p = pres*(pres <= 200)
    ## mode for 200 - 1000
    # beta1_p = pres*(pres > 200)*(pres <= 1000)
    ## mode for 500-1000
    # beta2_p = pres*(pres > 1000)
    ## constant pressure
    # const_p = np.ones(pres.shape[0])

    # Create a processed dataset #
    region['alpha0_x'] = alpha0_x; region['alpha1_x'] = alpha1_x; region['sin_days'] = sin_days; region['cos_days'] = cos_days
    region['lat_sin'] = lat_sin; region['lon_sin'] = lon_sin; region['lon_cos'] = lon_cos
    # region['beta0_p'] = beta0_p; region['beta1_p'] = beta1_p; region['beta2_p'] = beta2_p

    data_split(region, fe)


def data_split(region, fe = 0.1):
  """
  Prepare the data to feed the model

  Parameters:
  ----------
  - region: dict
  - fe: float (default = 0.1)
      percentage of data to feed the model

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

  for i,uni in enumerate(uniq_profile):

      index = np.where(region['profile_ids'] == uni)[0]
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

  sav_obj = open("x_train_s.pkl", 'wb')
  pickle.dump(X_train,sav_obj)
  sav_obj.close()

  sav_obj = open("y_train_s.pkl", 'wb')
  pickle.dump(y_train,sav_obj)
  sav_obj.close()

  sav_obj = open("feature_train_s.pkl", 'wb')
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


def init_EM(X,Y,K,method):

  # Sample size, number of features
  N=size(Y,0)
  Nb_features = X.shape[1]
  # Targets
  tars = Y.shape[1]

  # Random class assignments
  if method == "kmeans":
      clust = KMeans(n_clusters=K).fit_predict(X)

  elif method == "random":
      clust = np.random.randint(0,K, size = N);

  # Initialize the lambda
  hist_k = plt.hist(clust,K,density=True);
  plt.close()
  width_bins = hist_k[1][1] -hist_k[1][0]
  nb_k = hist_k[0]*width_bins * len(X)

  lambda_init=zeros(K)
  for k in range(K):
      lambda_init[k]=nb_k[k]/N

  Beta_init  =array(zeros((K,Nb_features,tars)))
  Sigma_init =array(zeros((K,tars,tars)))

  # Initialize Beta and Sigma
  for k in range(0,K):
      Beta_init[k,:,:] = np.linalg.lstsq(X[np.where(clust==k)[0],:],Y[np.where(clust==k)[0],:])[0]
      Sigma_init[k,:,:] = cov((Y[np.where(clust==k)[0],:]-X[np.where(clust==k)[0],:]@(Beta_init[k,:,:].reshape((Nb_features,tars)))).T)

  return(lambda_init,Beta_init,Sigma_init)


def EM_GPU(X_np,Y_np,lambda_init,Beta_init,Sigma_init,iter_EM, print_ = False):
  """
  Implement the EM algorithm on GPU.

  Returns:
  --------
  - log_lik: list
      Logarithm of likelihoods of each EM iteration (size = iter_EM)
  - lambda_hat: array
      The posterior propabilities of the latent variables (size =(nb of classes,))
  - Beta_hat: array
     ...  size = (nb of classes,nb_features,targets)
  - Sigma_hat: array
     ...  size = (nb of classes,targets,targets)
  - pi_hat: array
      The prior probabilities of the latent variables (size = (samples, nb of classes))
  - Y_hat: array
      The estimated targets (size = (samples,targets))
  - Z_hat: array
      The maximum likelihood of each samples (size =(samples,1))
  """
  # Number of classes
  K = len(Beta_init)

  # Number of samples, and targets
  n = Y_np.shape[0]
  tars = Y_np.shape[1]

  # Number of features
  nb_features = X_np.shape[1]

  # Initialization, convert to the cuda in Pytorch
  X = torch.from_numpy(X_np).cuda()
  Y = torch.from_numpy(Y_np).cuda()
  lambda_hat = torch.from_numpy(lambda_init).cuda()
  Beta_hat   = torch.from_numpy(Beta_init).cuda()
  Sigma_hat  = torch.from_numpy(Sigma_init).cuda()

  log_lik =[]
  pi_hat = torch.zeros(n,K).cuda()
  Z_hat = torch.zeros(1,n).cuda()

  if print_:
      display(["***EM_latent_class_regression***"]);

########## Start the EM algorithm ...
  for iE in range(iter_EM):
      ########## E-step
      sum_prob = torch.zeros(n,1).cuda()
      for k in range(K):
          # Compute the sum of products of the likelihood with a class's prior for each profile (formula 5)
          sum_prob = sum_prob + lambda_hat[k]*torch.exp(MultivariateNormal(X@Beta_hat[k,:,:],Sigma_hat[k,:,:]).log_prob(Y)).reshape(n,1)

      for k in range(K):
          # Compute how many percentage each class representing in each profile (formla 7)
          pi_hat[:,k]= lambda_hat[k]*torch.exp(MultivariateNormal(X@Beta_hat[k,:,:],Sigma_hat[k,:,:]).log_prob(Y))/sum_prob[:,0]

      # Maximum likelihood of each example
      Z_hat = torch.argmax(pi_hat, dim=1).reshape(1,n)
      # Update the lamda hat (formula 8)
      lambda_hat = torch.sum(pi_hat,0)/n

      ########## M-step
      for k in range(K):
          # Beta_hat[k,:,:]= torch.lstsq(torch.mul(Y,pi_hat[:,k].reshape(n,1)),torch.mul(X,pi_hat[:,k].reshape(n,1)))[0][nb_features,:]
          Beta_hat[k,:,:]= torch.inverse(X.T@torch.mul(pi_hat[:,k].reshape(n,1),X))@X.T@torch.mul(pi_hat[:,k].reshape(n,1),Y)
          # Update sigma (formula - 9)
          Sigma_hat[k,:,:]= torch.mul(pi_hat[:,k].reshape(n,1),Y-X@Beta_hat[k,:,:]).T@(Y-X@Beta_hat[k,:,:])/torch.sum(pi_hat[:,k],0)

      # Stock the likelihood of each sample for an iteration
      lik_tmp = torch.zeros(n,1).cuda()
      for k in range(K):
        lik_tmp = lik_tmp + lambda_hat[k]*torch.exp(MultivariateNormal(X@Beta_hat[k,:,:],Sigma_hat[k,:,:]).log_prob(Y)).reshape(n,1)

      log_lik_tmp_np = torch.sum(torch.log(lik_tmp),0).cpu().numpy()
      log_lik.append(log_lik_tmp_np)
########## Finish the EM algorithm !

  # Generate the estimated targets
  Y_hat=torch.zeros(n,tars).cuda();
  for k in range(K):
    Y_hat = Y_hat + torch.mul(pi_hat[:,k].reshape(n,1),X@Beta_hat[k,:,:])

  # Convert to the Numpy type
  Y_hat_np = Y_hat.cpu().numpy()
  lambda_hat_np = lambda_hat.cpu().numpy()
  Beta_hat_np =  Beta_hat.cpu().numpy()
  Sigma_hat_np = Sigma_hat.cpu().numpy()
  pi_hat_np = pi_hat.cpu().numpy()
  Z_hat_np = Z_hat.cpu().numpy()

  return(log_lik,lambda_hat_np,Beta_hat_np,Sigma_hat_np,Y_hat_np,pi_hat_np,Z_hat_np)


def BIC(inputs_):
  """
  BIC criteria calculation

  Return:
  ------
  - BIC_: float
      model's BIC
  """
  X,Y,nb_class,method,iter_EM = inputs_

  # Initiate the EM algorithm
  lambda_init,Beta_init,Sigma_init = init_EM(X,Y,nb_class,method)
  # Peform the EM algorithm
  log_lik,lambda_hat,Beta_hat,Sigma_hat,Y_hat,pi_hat,Z_hat=EM_GPU(X,Y,lambda_init,Beta_init,Sigma_init,iter_EM)
  # print('Beta_hat: ',Beta_hat)
  # print('Sigma_hat: ',Sigma_hat)

  # Sample size
  N = X.shape[0]
  # Number of parameters
  Nb_params = Sigma_hat.shape[0]*Sigma_hat.shape[1]*Sigma_hat.shape[2] + Beta_hat.shape[0]*Beta_hat.shape[1]*Beta_hat.shape[2]

  # Calculate the BIC of the model, in the last EM iteration
  last_log_lik = log_lik[-1][0]
  BIC_ = -2*last_log_lik + Nb_params*np.log(N)
  return BIC_,log_lik,Nb_params,Beta_hat,Sigma_hat,lambda_hat
