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
import torch
import julian
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
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
    plt.ylabel('Predicted Temperature')
    plt.xlabel('GT Temperature')
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
    for k in range(K):
        plt.scatter(days_mean[:-2],prior_prob_days_mean[k,:-2])
        plt.plot(days_mean[:-2],prior_prob_days_mean[k,:-2], label = 'Mode-'+str(k+1))
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

def pcolor_surface(lon, lat, Nlon, Nlat, map,temp, title, combine = False, subplot = 221):
    """
    pcolor plot for sea surface temperature

    - Nlon: number of point discritizing the longitude
    - Nlat: number of point discritizing the latitude
    """

    xlat = np.linspace(min(lat),max(lat),Nlat)
    xlon = np.linspace(min(lon),max(lon),Nlon)

    Temp = np.zeros((xlat.shape[0],xlon.shape[0]))
    Freq = np.zeros((xlat.shape[0],xlon.shape[0]))

    N = lon.shape[0]

    for k in range(N):
        i = np.argmin(abs(xlat - lat[k]))
        j = np.argmin(abs(xlon - lon[k]))
        Temp[i,j] += temp[k]
        Freq[i,j] += 1

    if combine:
        plt.subplot(subplot)
    ax = plt.gca()
    temp_avg = np.divide(Temp,Freq)
    plon, plat = map(xlon, xlat)
    xxlon,xxlat = meshgrid(plon,plat)
    cmap = 'coolwarm'
    map.contourf(xxlon, xxlat, temp_avg, cmap = cmap)
    plt.title(title)
    map.drawcoastlines()
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


def follow_x_plot(daysx,tempx,prof,step = 1):
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
    plt.title("Sea Surface Temperature Evolution ");
    plt.ylabel("Temperature C")
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
    colors = ['tab:red','tab:green','tab:orange','tab:blue','tab:purple']

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
