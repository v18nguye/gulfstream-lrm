import pickle as pk
import pandas as pd
import numpy as np
from numpy import *
import time
import matplotlib.pyplot as plt
from tqdm import *
from pylab import *
from multiprocessing import Pool
import pickle
import os
import statsmodels.api as sm
import torch
import julian
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
from scipy.interpolate import griddata
from scipy.stats import multivariate_normal as mn
from matplotlib.pyplot import figure
#import torch
from torch.distributions.multivariate_normal import MultivariateNormal
nasa_julian = 98
cnes_julian = 90
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score


def y_hat_esti(X_,y_,Beta,Sigma,Lambda,n):
  """ Estimate y_hat

  - Beta: (K,nb_fea,nb_tar)
  - Sigma: (K,nb_tar,nb_tar)
  - Lambda: (K,1)
  - n: nb of samples
  """

  K  = Beta.shape[0]
  tars = Beta.shape[2]

  X = torch.from_numpy(X_).cuda()
  Y = torch.from_numpy(y_).cuda()
  Beta_hat = torch.from_numpy(Beta).cuda()
  Sigma_hat = torch.from_numpy(Sigma).cuda()
  Lambda_hat = torch.from_numpy(Lambda).cuda()

  pi_hat = torch.zeros(n,Lambda.shape[0]).cuda()
  y_hat = torch.zeros(n,tars).cuda()
  sum_prob = torch.zeros(n,1).cuda()

  for k in range(K):
      # Compute the sum of products of the likelihood with a class's prior for each profile (formula 5)
      sum_prob = sum_prob + Lambda_hat[k]*torch.exp(MultivariateNormal(X@Beta_hat[k,:,:],Sigma_hat[k,:,:]).log_prob(Y)).reshape(n,1)

  for k in range(K):
    # Compute how many percentage each class representing in each profile (formla 7)
    pi_hat[:,k]= Lambda_hat[k]*torch.exp(MultivariateNormal(X@Beta_hat[k,:,:],Sigma_hat[k,:,:]).log_prob(Y))/sum_prob[:,0]

  for k in range(K):
    y_hat = y_hat + torch.mul(pi_hat[:,k].reshape(n,1),X@Beta_hat[k,:,:])

  y_hat_np = y_hat.cpu().numpy()
  pi_hat_np = pi_hat.cpu().numpy()

  return y_hat_np,pi_hat_np


def likli_plot(log_likli,coln,row,nb_class,nb_iter_,iter_EM):
  """ plot the log likelihood of models
  """
  figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
  for cla_ in range(len(nb_class)):
    plt.subplot(row, coln, cla_ + 1)
    for iter_ in range(nb_iter_):
      plt.plot(range(iter_EM),np.asarray(log_likli[cla_,iter_,:]))
      plt.ylabel('Log likelihood')
      plt.xlabel('EM iterations')
      plt.title('Model with:  '+str(nb_class[cla_])+' classes')


def bic_box(bic,nb_class):
  """ plot box plot of model's bics
  """
  fig = plt.figure(figsize=(7,7))
  ax = fig.add_axes([1,1,1,1])
  labels = [min(nb_class)+i for i in range(len(nb_class))]
  plt.boxplot(bic, labels  = labels)
  plt.title("Boxplot BIC")
  xlabel("Nb_of_class")
  ylabel("BIC")


def extract_bsl(beta_,sigma_,lambda_,X,y,nb_class,th_class,nb):
  """
    Etract beta, sigma, lambda of a specific class

    - th_class: the class number extracted
    - nb th experience

  """

  index = 0
  th = 0

  for c in range(len(nb_class)):
    beta_c = beta_[c,index:index + nb_class[th],:,:]
    sigma_c = sigma_[c,index:index + nb_class[th],:,:]
    lambda_c = lambda_[c,index:index + nb_class[th],:]

    if nb_class[c] == th_class:

      beta = open('beta_E'+str(nb)+'_'+str(th_class)+'.pkl', 'wb')
      pickle.dump(beta_c, beta)
      sigma = open('sigma_E'+str(nb)+'_'+str(th_class)+'.pkl', 'wb')
      pickle.dump(sigma_c, sigma)
      lambda_ = open('lambda_E'+str(nb)+'_'+str(th_class)+'.pkl', 'wb')
      pickle.dump(lambda_c, lambda_)
      beta.close()
      sigma.close()
      lambda_.close()
      return  beta_c,sigma_c,lambda_c
      break

    index = index + nb_class[th]
    th = th +1


def y_eval(beta_,sigma_,lambda_,X,y,nb_class,coln,row, save,nb, test):

  """
    Evaluate the y and  its estimation

  - save: the class saved for result visualization
  - test: if test True, save as test result
  - nb: nb th experience
  """

  index = 0
  th = 0

  figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
  x_linsp = np.linspace(-5,35,50)
  markers = 0.1*np.full((y.shape[0], 1), 1)
  for c in range(len(nb_class)):
    beta_c = beta_[c,index:index + nb_class[th],:,:]
    sigma_c = sigma_[c,index:index + nb_class[th],:,:]
    lambda_c = lambda_[c,index:index + nb_class[th],:]
    # y_esti,_ = y_hat_esti(X,y,beta_c,sigma_c,lambda_c,X.shape[0])

    # save interesting class
    if nb_class[c] == save:
        y_esti,_ = y_hat_esti(X,y,beta_c,sigma_c,lambda_c,X.shape[0]) ## new
        if test:
            np.savetxt('y_h_test_E'+str(nb)+'_'+str(nb_class[c])+'.txt',y_esti)
        else:
            np.savetxt('y_h_train_E'+str(nb)+'_'+str(nb_class[c])+'.txt',y_esti)

    # plt.subplot(row, coln, c+1)
    #
    # plt.scatter(y,y_esti, marker = ".", s = markers, c ='b', label = "r2_score = " +str(r2_score(y, y_esti)))
    # plt.plot(x_linsp,x_linsp,'k')
    # plt.ylabel('Predicted Temperature')
    # plt.xlabel('GT Temperature')
    # plt.title('Model with: '+str(nb_class[c])+' classes')
    # plt.legend()
    index = index + nb_class[th]
    th = th +1


def sal_eval(beta_,sigma_,lambda_,X,y,nb_class,coln,row, save,nb, test,  fx1 =30 , fx2 = 40):

  """
    Evaluate the y and  its estimation

  - save: the class saved for result visualization
  - test: if test True, save as test result
  - nb: nb th experience
  """

  index = 0
  th = 0

  figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')
  x_linsp = np.linspace(fx1,fx2,50)
  markers = 0.1*np.full((y.shape[0], 1), 1)
  for c in range(len(nb_class)):
    beta_c = beta_[c,index:index + nb_class[th],:,:]
    sigma_c = sigma_[c,index:index + nb_class[th],:,:]
    lambda_c = lambda_[c,index:index + nb_class[th],:]
    y_esti,_ = y_hat_esti(X,y,beta_c,sigma_c,lambda_c,X.shape[0])

    # save interesting class
    if nb_class[c] == save:
        if test:
            np.savetxt('y_h_test_E'+str(nb)+'_'+str(nb_class[c])+'.txt',y_esti)
        else:
            np.savetxt('y_h_train_E'+str(nb)+'_'+str(nb_class[c])+'.txt',y_esti)

    plt.subplot(row, coln, c+1)

    plt.scatter(y,y_esti, marker = ".", s = markers, c ='b', label = "r2_score = " +str(r2_score(y, y_esti)))
    plt.plot(x_linsp,x_linsp,'k')
    plt.ylabel('Predicted Salanity')
    plt.xlabel('GT Salanity')
    plt.title('Model with: '+str(nb_class[c])+' classes')
    plt.legend()
    index = index + nb_class[th]
    th = th +1


def pi_hat(X_,y_,Beta,Sigma,Lambda,n,nb,th_class, test):
  """ Estimate p_hat

  - Beta: (K,nb_fea,nb_tar)
  - Sigma: (K,nb_tar,nb_tar)
  - Lambda: (K,1)
  - n: nb of samples
  - nb: nb th experience
  - test : if test is true, save file as test
  """

  K  = Beta.shape[0]
  tars = Beta.shape[2]

  X = torch.from_numpy(X_).cuda()
  Y = torch.from_numpy(y_).cuda()
  Beta_hat = torch.from_numpy(Beta).cuda()
  Sigma_hat = torch.from_numpy(Sigma).cuda()
  Lambda_hat = torch.from_numpy(Lambda).cuda()

  pi_hat = torch.zeros(n,Lambda.shape[0]).cuda()
  y_hat = torch.zeros(n,tars).cuda()
  sum_prob = torch.zeros(n,1).cuda()

  for k in range(K):
      # Compute the sum of products of the likelihood with a class's prior for each profile (formula 5)
      sum_prob = sum_prob + Lambda_hat[k]*torch.exp(MultivariateNormal(X@Beta_hat[k,:,:],Sigma_hat[k,:,:]).log_prob(Y)).reshape(n,1)

  for k in range(K):
    # Compute how many percentage each class representing in each profile (formla 7)
    pi_hat[:,k]= Lambda_hat[k]*torch.exp(MultivariateNormal(X@Beta_hat[k,:,:],Sigma_hat[k,:,:]).log_prob(Y))/sum_prob[:,0]

  for k in range(K):
    y_hat = y_hat + torch.mul(pi_hat[:,k].reshape(n,1),X@Beta_hat[k,:,:])

  pi_hat_np = pi_hat.cpu().numpy()

  if test:
      np.savetxt('pi_h_test_E'+str(nb)+'_'+str(th_class)+'.txt',pi_hat_np)
  else:
      np.savetxt('pi_h_train_E'+str(nb)+'_'+str(th_class)+'.txt',pi_hat_np)


def prior_prob_time_plot(pi_hat,juld_test, K, m, title,figx =10, figy = 10):
    """
    - K: number of classes
    - m: size of the windows
    """
    encoded_juld = [julian.from_jd(round(x), fmt='mjd') for x in juld_test]
    days_in_year =  np.asarray([x.day + (x.month -1)*30 for x in encoded_juld])

    # prior probality per days
    prior_prob_days = np.zeros((K,365))

    days = np.linspace(1,365,365).astype(int)

    for day in days:
        day_mask = np.where(days_in_year == day)
        if size(day_mask) != 0:
            prior_prob_days[:,day-1] = mean(pi_hat[day_mask], axis = 0)

    mask = 1/m*np.ones(m)

    segment = int(365/len(mask))

    prior_prob_days_mean = np.zeros((K,segment))

    for k in range(segment):
        if (k == 0) or (k == segment-1):
            w = prior_prob_days[:,k*len(mask):(k+1)*len(mask)]
            prior_prob_days_mean[:, k] = w@mask
        else:
            w = prior_prob_days[:,k*len(mask) - int(len(mask)/2):k*len(mask)+len(mask) -int(len(mask)/2)]
            prior_prob_days_mean[:, k] = w@mask

    days_mean = np.asarray([i for i in range(segment)])*m
    x_labels = ['Jan','Feb','Mar','Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    spots = np.asarray([30*i for i in range(12)])


    figure(num=None, figsize=(figx, figy), dpi=80, facecolor='w', edgecolor='k')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for k in range(K):
        plt.scatter(days_mean[:-2],prior_prob_days_mean[k,:-2], color = colors[k])
        plt.plot(days_mean[:-2],prior_prob_days_mean[k,:-2], label = 'Mode-'+str(k+1), color=colors[k])
        plt.legend()
        plt.xticks(spots,x_labels);
        plt.ylabel('Priori Probability')
        plt.xlabel('Day')
        plt.title(title)
        plt.grid(True)


def temp_plot(lon_test,lat_test,map,temp):
    "Plot surface temperature"

    lons = lon_test
    lats = lat_test

    x, y = map(lons, lats)

    figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
    markers = np.full((lon_test.shape[0], 1), 1)
    ax = plt.gca()
#     map.scatter(x, y, c = temp, s = markers, cmap='coolwarm')
    map.scatter(x, y, c = temp, s = markers, cmap='Greys')
    map.drawcoastlines()
    # plot colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)

def follow_x(index_,coords_g,priode_,X,f_x,std_x,pi_hat,beta,lambda_,sigma,gt_temp):
    "Following the temperature evolution of a point x"

    index = index_
    latx, lonx, juldx0 = coords_g[index,0],coords_g[index,1],coords_g[index,2]
    print("Coordinates x: ","(longtitude= "+str(lonx)+",latitude= "+str(latx)+")")
    x0 = copy.deepcopy(X[index,:])
    priode = priode_
    tempx = []
    tempx.append(gt_temp[index_])
    daysx = []
    daysx.append(juldx0)
    pi_hatx0 = copy.deepcopy(pi_hat[index,:])
    pi_hatx = pi_hatx0
    K = pi_hatx.shape[0]

    for j in range(1,priode):

        juldx = juldx0 + j

        # update date of x
        x0[1] = juldx
        x0[2] = np.sin(2*math.pi*(1/365.25)*juldx)
        x0[3] = np.cos(2*math.pi*(1/365.25)*juldx)

        # normalize x
        x = (x0 - f_x)/std_x

        # update prior mode probabilities of x
        p = 0
        for k in range(K):
            p += lambda_[k]*mn.pdf(tempx[j-1],x@beta[k,:,:],sigma[k,:,:])

        for k in range(K):
            pi_hatx[k]= lambda_[k]*mn.pdf(tempx[j-1],x@beta[k,:,:],sigma[k,:,:])/p

        yx = 0
        # estimate temperature
        for λ in range(pi_hatx.shape[0]):
            yx = yx + pi_hatx[λ]*x@beta[λ,:,:].ravel()

        daysx.append(juldx)
        tempx.append(yx)

    return daysx,tempx


def follow_x_plot(daysx,tempx,prof,step = 1,  temp = True):
    """
    Plot evolution of sea surface temperature at a specific coordinate

    + step: step plot for year info
    """

    years = np.asarray([julian.from_jd(d, fmt='mjd').year + cnes_julian for d in list(daysx)])
    u_year, c_year = np.unique(years,return_counts = True)
    u_year = list(u_year)
    u_year[0] = str(julian.from_jd(daysx[0], fmt='mjd').year + cnes_julian)+"-"+str(julian.from_jd(daysx[0], fmt='mjd').month)+"-"+str(julian.from_jd(daysx[0], fmt='mjd').day)
    spots_ = np.asarray([sum(c_year[:i]) for i in range(c_year.shape[0])])
    spots = spots_[[k*step for k in range(int(spots_.shape[0]/step))]]
    fig, ax = plt.subplots(figsize=(15, 6))
    plot(np.asarray(tempx), label = "Profile: "+str(prof));
    plt.xticks(spots,u_year);
    if temp == True:
        plt.title("Sea Surface Temperature Evolution ");
        plt.ylabel("Temperature C")
        plt.xlabel("Year")
    else:
        plt.title("Sea Surface Salanity Evolution ");
        plt.ylabel("Salanity")
        plt.xlabel("Year")
    plt.grid(True)
    plt.legend()


def mode_dist(lon_test,lat_test,map,pi_hat, title, combine = False, subplot = 221):
    "Plot dynamical mode distributions"

    lons = lon_test
    lats = lat_test
    x, y = map(lons, lats)

    dominant_mode = np.argmax(pi_hat, axis = 1)

    modes = np.unique(dominant_mode)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    if combine:
        plt.subplot(subplot)

    for m in list(modes):
        mask = np.where(dominant_mode == m)
        x_mask = x[mask]
        y_mask = y[mask]
        markers = np.full((len(mask), 1), 1)
        dominants = dominant_mode[mask]
        map.scatter(x_mask, y_mask, c = colors[m], label = "mode"+str(m+1), s = markers)
    map.drawcoastlines()
    plt.legend()
    plt.title(title)


def mode(lon_test,lat_test,map,pi_hat, m, title, combine = False, subplot = 221):
    "Plot dynamical mode distributions"

    lons = lon_test
    lats = lat_test
    x, y = map(lons, lats)

    if combine:
        plt.subplot(subplot)

    markers = np.full((len(pi_hat), 1), 1)
    map.scatter(x, y, c = pi_hat, label = "mode"+str(m+1), s = markers)
    map.drawcoastlines()
    plt.legend()
    plt.title(title)
#########################################
## Plot spatial temperature distribution#
#########################################
def target_predict(lon, lat, pres, Nlon, Nlat,map, gt_targ, est_targ, error, pres_lev, neigboors, cmap_temp = True):
    """
    target prediction in time and space

    - lat: latitude of the data
    - lon: longitude of the data
    - neigboors: for selecting the interpolated region
    - Nlon: number of point discritizing the longitude
    - Nlat: number of point discritizing the latitude
    - pres_lev: a determined pressure at which we present data ± 15
    """

    # create a 2D coordinate grid
    xlat = np.linspace(min(lat),max(lat),Nlat)
    ylon = np.linspace(min(lon),max(lon),Nlon)

    yylon, xxlat = np.meshgrid(ylon,xlat)
    xxlat = xxlat.reshape(xxlat.shape[0]*xxlat.shape[1],1)
    yylon = yylon.reshape(xxlat.shape[0]*xxlat.shape[1],1)
    zzdepth = pres_lev*np.ones((xxlat.shape[0]*xxlat.shape[1],1))

    # interpolation on the 2D grid at the pressure = pres_lev
    mask_lev = (pres_lev-15 < pres)*(pres < pres_lev + 15)

    lev_pres = pres[mask_lev]
    lev_lat = lat[mask_lev]
    lev_lon = lon[mask_lev]
    lev_gt = gt_targ[mask_lev]
    lev_est = gt_targ[mask_lev]
    lev_error = error[mask_lev]

    N = lev_lon.shape[0]
    features = np.concatenate((lev_lat.reshape(N,1),lev_lon.reshape(N,1),lev_pres.reshape(N,1)),axis = 1)
    points = np.concatenate((xxlat,yylon,zzdepth),axis = 1)

    gt_interpol = griddata(features,lev_gt,points, rescale=True)
    gt_interpol = gt_interpol.reshape(Nlat,Nlon)

    est_interpol = griddata(features,lev_est,points, rescale=True)
    est_interpol = est_interpol.reshape(Nlat,Nlon)

    err_interpol = griddata(features,lev_error,points, rescale=True)
    err_interpol = err_interpol.reshape(Nlat,Nlon)

    err_mean = np.abs(lev_error).mean()

    # select interpolated region where the data exists
    mask_pres = pres < 30
    pres_under_30 = pres[mask_pres]
    lon_30 = lon[mask_pres]
    lat_30 = lat[mask_pres]
    # existing data region mask (care  only about the surface)
    mask_region = np.ones((Nlat,Nlon)) < 2
    for k in range(len(lon_30)):
        i = np.argmin(abs(xlat - lat_30[k]))
        j = np.argmin(abs(ylon - lon_30[k]))
        mask_region[i-neigboors:i+neigboors,j-neigboors:j+neigboors] = False
    # apply the mask
    gt_interpol[mask_region] = nan
    est_interpol[mask_region] = nan
    err_interpol[mask_region] = nan

    if cmap_temp:

        cmap = 'coolwarm'
    else:
        cmap = 'jet'

    plon, plat = map(ylon, xlat)
    xxlon,xxlat = meshgrid(plon,plat)

    parallels = np.arange(0.,81,5.) # lat
    meridians = np.arange(10.,351.,5.) # lon

    fig  = plt.figure(figsize = (15,10))
    subplots_adjust(wspace = 0.1, hspace = 0.2)

    ax1 = fig.add_subplot(221)
    map.contourf(xxlon, xxlat, gt_interpol, cmap = cmap)
    plt.title("GT-PRES-%.f"%pres_lev)
    map.drawcoastlines()
    map.drawparallels(parallels,labels=[True,False,True,False],linewidth=0.3);
    map.drawmeridians(meridians,labels=[True,False,False,True],linewidth=0.3);
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax1)

    ax2 = fig.add_subplot(222)
    map.contourf(xxlon, xxlat, est_interpol, cmap = cmap)
    plt.title("EST-PRES-%.f"%pres_lev)
    map.drawcoastlines()
    map.drawparallels(parallels,labels=[True,False,True,False],linewidth=0.3);
    map.drawmeridians(meridians,labels=[True,False,False,True],linewidth=0.3);
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax2)

    ax3 = subplot2grid((2,8), (1, 2), colspan=4)
    map.contourf(xxlon, xxlat, err_interpol, cmap = cmap)
    plt.title("%.f-Global-Error-%.2f °C"%(pres_lev,err_mean))
    map.drawcoastlines()
    map.drawparallels(parallels,labels=[True,False,True,False],linewidth=0.3);
    map.drawmeridians(meridians,labels=[True,False,False,True],linewidth=0.3);
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax3)

###################################
## Plot spatial mode distribution #
###################################
def spa_dyna_mode_dis(pi_hat, lon, lat, Nlat, Nlon, pres, h_depth,l_depth, inter_depth,neigboors, map):
    """
    Plot the spatial dynamical mode distribution between two depth levels in the ocean.

    Args:
    -----
    - pi_hat: the prior probabilities for dynamical modes
    - lon: ...
    - lat: ...
    - pres: infos of the pressure at each point in the ocean
    - h_depth: the higher depth level from the inter_depth
    - l_depth: the lower depth from the the inter_depth
    - inter_depth: a depth level at which we interpolate the prior dynamical modes
    - map: a basemap object
    """

    ## create a 2D coordinate grid
    xlat = np.linspace(min(lat),max(lat),Nlat)
    ylon = np.linspace(min(lon),max(lon),Nlon)

    yylon, xxlat = np.meshgrid(ylon,xlat)
    xxlat = xxlat.reshape(xxlat.shape[0]*xxlat.shape[1],1)
    yylon = yylon.reshape(xxlat.shape[0]*xxlat.shape[1],1)
    zzdepth = inter_depth*np.ones((xxlat.shape[0]*xxlat.shape[1],1))

    ## Extract data
    data_mask = np.where((pres >= h_depth)*(pres <= l_depth))
    extracted_priors = pi_hat[data_mask[0],:][:10000,:]
    lev_lon = lon[data_mask[0]][:10000]
    lev_lat = lat[data_mask[0]][:10000]
    lev_pres = pres[data_mask[0]][:10000]

    N = lev_lon.shape[0]
    features = np.concatenate((lev_lat.reshape(N,1),lev_lon.reshape(N,1),lev_pres.reshape(N,1)),axis = 1)
    points = np.concatenate((xxlat,yylon,zzdepth),axis = 1)

    ## existing data region
    mask_pres = pres < 30
    pres_under_30 = pres[mask_pres]
    lon_30 = lon[mask_pres]
    lat_30 = lat[mask_pres]
    M = extracted_priors.shape[1]

    # interpolation
    dyn_interpols = np.zeros((Nlat*Nlon,M))
    for dyn in tqdm(range(M), disable=False):
        dyn_interpols[:,dyn] = griddata(features,extracted_priors[:,dyn],points, rescale=True)
        dyn_interpols[np.isnan(dyn_interpols[:,dyn]),dyn] = 0
    dyn_interpols = dyn_interpols/(dyn_interpols.sum(axis =1).reshape(Nlat*Nlon,1))


    fig = plt.figure(figsize = (20,15))
    subplots_adjust(wspace=0.2, hspace=0.2)

    for dyn in tqdm(range(M), disable=False):
        dyn_interpol = dyn_interpols[:,dyn].reshape(Nlat,Nlon)

        # existing data region mask (care  only about the surface)
        mask_region = np.ones((Nlat,Nlon)) < 2
        for k in range(len(lon_30)):
            i = np.argmin(abs(xlat - lat_30[k]))
            j = np.argmin(abs(ylon - lon_30[k]))
            mask_region[i-neigboors:i+neigboors,j-neigboors:j+neigboors] = False
        # apply the mask
        dyn_interpol[mask_region] = nan

        cmap = 'YlOrBr'

        plon, plat = map(ylon, xlat)
        xxlon_,xxlat_ = meshgrid(plon,plat)

        parallels = np.arange(0.,81,5.) # lat
        meridians = np.arange(10.,351.,5.) # lon

        if dyn == 6:
            ax = subplot2grid((3,12), (2, 1), colspan=4)
            map.contourf(xxlon_, xxlat_, dyn_interpol, cmap = cmap)
            plt.title("Dyn-Mode-"+str(dyn+1))
            map.drawcoastlines()
            map.drawparallels(parallels,labels=[True,False,True,False],linewidth=0.3);
            map.drawmeridians(meridians,labels=[True,False,False,True],linewidth=0.3);
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cax=cax)

        elif dyn == 7:
            ax = subplot2grid((3,12), (2, 6), colspan=4)
            map.contourf(xxlon_, xxlat_, dyn_interpol, cmap = cmap)
            plt.title("Dyn-Mode-"+str(dyn+1))
            map.drawcoastlines()
            map.drawparallels(parallels,labels=[True,False,True,False],linewidth=0.3);
            map.drawmeridians(meridians,labels=[True,False,False,True],linewidth=0.3);
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cax=cax)

        else:
            plt.subplot(3,3,dyn+1)
            ax = plt.gca()
            map.contourf(xxlon_, xxlat_, dyn_interpol, cmap = cmap)
            plt.title("Dyn-Mode-"+str(dyn+1))
            map.drawcoastlines()
            map.drawparallels(parallels,labels=[True,False,True,False],linewidth=0.3);
            map.drawmeridians(meridians,labels=[True,False,False,True],linewidth=0.3);
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cax=cax)
###############################
## Plot the seasonal variation#
###############################

def sea_temp(indexs, gt_temps, est_temps, gt_dates):
    """
    Calulate the  seasonal temperature variation in a small specific zone

    Args:
    - indexs: indexs of related points
    - gt_temps: the ground-truth temperature
    - est_temps: the estimated temperature

    """
    gt = gt_temps[indexs]
    est = est_temps[indexs]
    residus = np.abs(gt-est)
    dates = gt_dates[indexs]

    sorted_date_indexes = np.argsort(dates)
    sorted_dates = dates[sorted_date_indexes]
    sorted_gt = gt[sorted_date_indexes]
    sorted_est = est[sorted_date_indexes]
    sorted_residus = residus[sorted_date_indexes]

    jdays = [julian.from_jd(x, fmt='mjd') for x in sorted_dates]
    days = np.array([x.day + (x.month -1)*30 for x in jdays])

    _, trend_gt = sm.tsa.filters.hpfilter(sorted_gt, 1000)
    _, trend_est = sm.tsa.filters.hpfilter(sorted_est, 1000)
    _, trend_residus = sm.tsa.filters.hpfilter(sorted_residus, 500)

    temps_in_year_gt = np.zeros(365)
    temps_in_year_est = np.zeros(365)
    temps_in_year_residus = np.zeros(365)

    for d in range(0,365):

        temps_in_year_gt[d] = np.mean(trend_gt[np.where(days == d +1)])
        temps_in_year_est[d] = np.mean(trend_est[np.where(days == d +1)])
        temps_in_year_residus[d] = np.std(trend_residus[np.where(days == d +1)])

        if np.isnan(temps_in_year_gt[d]):
            temps_in_year_gt[d] = temps_in_year_gt[d-1]
        if np.isnan(temps_in_year_est[d]):
            temps_in_year_est[d] = temps_in_year_est[d-1]
        if np.isnan(temps_in_year_residus[d]):
            temps_in_year_residus[d] = temps_in_year_residus[d-1]

    return temps_in_year_gt, temps_in_year_est, temps_in_year_residus


def sea_temp_plot(p_index,coords,X, gt_temp, est_temp, t1, t2, t3, lat_thres = 3,lon_thres = 3, date_thres = 1000, depth_thres = 10, combine = True ,subplot = 221):
    """
    Seasonal temperature visualization at severals small local regions in ocean
    Example: sea_temp_plot(p_index,coords_g_train,X, gt_temp_train, est_temp_train,t1 = 1000,t2 = 1000, t3 =50,
    lat_thres = 3,lon_thres = 3, date_thres = 4745, depth_thres = 10, combine = True ,subplot = 221)
    Args:
    -----
    - p_index: index of the point in data
    - coords: containing lat,lon, juld infos
    """

    # extract local points
    lat = coords[:,0]
    lon = coords[:,1]
    juld = coords[:,2]
    indexs = index_extract(p_index,lat,lon,juld,X,lat_thres = lat_thres, lon_thres = lon_thres, date_thres = date_thres, depth_thres = depth_thres, patch = True)

    gt_trend, est_trend, residu_trend = sea_temp(indexs, gt_temp, est_temp, juld)

    # apply hp filters
    _, trend_gt = sm.tsa.filters.hpfilter(gt_trend, t1)
    _, trend_est = sm.tsa.filters.hpfilter(est_trend, t2)
    _, trend_residu = sm.tsa.filters.hpfilter(residu_trend, t3)

    low_err = trend_est - trend_residu
    high_err = trend_est + trend_residu

    x_labels = ['Jan','Feb','Mar','Apr','May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    spots = np.asarray([30*i for i in range(12)])

    if combine:
        plt.subplot(subplot)

    plt.plot(trend_gt, label = 'GT Trend')
    plt.plot(trend_est, label = 'Est Trend')
    plt.fill_between(np.linspace(0,364,365),low_err,high_err,color='green', alpha=0.2, label = 'Std-Error')
    plt.title('LAT= %.2f LON= %.2f PRES= %.2f'%(coords[p_index,0],coords[p_index,1],X[p_index,-1]))
    plt.xlabel('Months')
    plt.ylabel('Temperature')
    plt.xticks(spots,x_labels)
    plt.legend()

def index_extract(p_index,lat,lon,juld,X,lat_thres, lon_thres, date_thres, depth_thres, patch = True):
    """
    Extract indexes in specific conditions
    """

    depth_point = X[:,-1][p_index]
    lat_point  = lat[p_index]
    lon_point = lon[p_index]
    date_point = juld[p_index]

    depth_mask = np.abs(X[:,-1] - depth_point) < depth_thres
    lat_mask =  np.abs(lat - lat_point) <lat_thres
    lon_mask = np.abs(lon - lon_point) < lon_thres
    date_mask = np.abs(juld - date_point) < date_thres

    index = np.where(depth_mask*lat_mask*lon_mask*date_mask)[0]
    return index
