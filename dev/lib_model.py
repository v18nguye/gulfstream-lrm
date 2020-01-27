import pickle as pk
import pandas as pd
import numpy as np
from numpy import *
import time
from tqdm import *
from pylab import *
from multiprocessing import Pool
import pickle
import scipy.stats
from scipy.linalg import qr, solve, lstsq
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.pyplot import figure

def input_data(X, y, min_class, max_class, iter_EM_, method_, nb_inter_):

    """
    - X: samples
    - y: target
    - min_class: min of number of class
    - max_class: max of number of class
    - nb_class: a list of nb of classes
    - iter_EM_: number of iterations for the EM algorithm
    - method: initial method
    - nb_inter_: nb of tests for each class
    """

    start = time.time()
    nb_class=[i for i in range(min_class,max_class+1)]
    len_class =len(nb_class)

    inputs=[]
    iter_EM= iter_EM_
    method = method_
    nb_iter = nb_inter_
    for i in range(len_class):
        for j in range(nb_iter):
            inputs.append([X,y,nb_class[i],method,iter_EM])

    return nb_class, inputs


def data_stdliz(X):
  """ Standardlize the data

  X: sample

  """
  X_new = (X - mean(X, axis=0)) / std(X, axis=0)

  return X_new


def init_EM(X,Y,K,method):

  # Sample size, number of features
  N=size(Y,0)
  Nb_features = X.shape[1]
  n = X.shape[0]
  # Targets
  tars = Y.shape[1]

  # Random class assignments
  if method == "kmeans":
      clust = KMeans(n_clusters=K).fit_predict(np.hstack((X,Y)))

  elif method == "random":
      clust = np.random.randint(0,K, size = N);

  # Initialize the lambda
  uni_clus,counts = np.unique(clust, return_counts = True)

  lambda_init=zeros(K)
  for k in range(K):
      lambda_init[k]=counts[k]/N

  pi_hat = np.zeros((clust.size,clust.max()+1))
  pi_hat[np.arange(clust.size),clust] = 1

  Beta_init  =array(zeros((K,Nb_features,tars)))
  Sigma_init =array(zeros((K,tars,tars)))

  # Initialize Beta and Sigma
  for k in range(0,K):
    Beta_init[k,:,:] = np.linalg.inv(X.T@np.multiply(pi_hat[:,k].reshape(n,1),X))@X.T@np.multiply(pi_hat[:,k].reshape(n,1),Y)
    Sigma_init[k,:,:] = np.multiply(pi_hat[:,k].reshape(n,1),Y-X@Beta_init[k,:,:]).T@(Y-X@Beta_init[k,:,:])/sum(pi_hat[:,k])

  return(lambda_init,Beta_init,Sigma_init)


def EM_GPU(X_np,Y_np,lambda_init,Beta_init,Sigma_init,iter_EM, print_ = True): ## Error fixed (9-1-20)
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

  X_cpu = X_np
  Y_cpu = Y_np
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

      ########## M-step
      # Update the lamda hat (formula 8)
      lambda_hat = torch.sum(pi_hat,0)/n
      # print("lambda_hat: ",lambda_hat)
      for k in range(K):
          # Beta_hat[k,:,:]= torch.lstsq(torch.mul(Y,pi_hat[:,k].reshape(n,1)),torch.mul(X,pi_hat[:,k].reshape(n,1)))[0][nb_features,:]
          Beta_hat[k,:,:]= torch.inverse(X.T@torch.mul(pi_hat[:,k].reshape(n,1),X))@X.T@torch.mul(pi_hat[:,k].reshape(n,1),Y)
          # Update sigma (formula - 9)
          Sigma_hat[k,:,:]= torch.mul(pi_hat[:,k].reshape(n,1),Y-X@Beta_hat[k,:,:]).T@(Y-X@Beta_hat[k,:,:])/torch.sum(pi_hat[:,k],0)
      # Stock the likelihood of each sample for an iteration
      lik_tmp = torch.zeros(n,1).cuda()

      for i in range(K):
        lik_tmp = lik_tmp + lambda_hat[i]*torch.exp(MultivariateNormal(X@Beta_hat[i,:,:],Sigma_hat[i,:,:]).log_prob(Y)).reshape(n,1)

      log_lik_tmp_np = torch.sum(torch.log(lik_tmp),0).cpu().numpy()
      log_lik.append(log_lik_tmp_np)
# ########## Finish the EM algorithm !

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


def BIC_GPU(inputs_):
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

  # Sample size
  N = X.shape[0]
  # Number of parameters
  Nb_params = Sigma_hat.shape[0]*Sigma_hat.shape[1]*Sigma_hat.shape[2] + Beta_hat.shape[0]*Beta_hat.shape[1]*Beta_hat.shape[2]

  # Calculate the BIC of the model, in the last EM iteration
  last_log_lik = log_lik[-1][0]
  BIC_ = -2*last_log_lik + Nb_params*np.log(N)
  return BIC_,log_lik,Nb_params,Beta_hat,Sigma_hat,lambda_hat


def model_test_GPU(inputs_train, nb_class, nb_iter_, iter_EM, nb_features, n_targets,test, save = True, al_targ= False):
  """ test models
  - inputs: list of inputs to model
  - nb_class: list of classes
  - nb_iter_: ...
  - iter_EM: ...
  - nb_features: ...
  - n_targets: ...
  - test: numero of the test
  - save: boolean
  - al_targ: boolean
  """

  len_class = len(nb_class)
  BIC_= []
  Log_lik = np.zeros((len_class,nb_iter_,iter_EM))
  Beta = np.zeros((len_class, np.sum(np.asarray(nb_class)), nb_features, n_targets))
  Sigma = np.zeros((len_class,np.sum(np.asarray(nb_class)), n_targets, n_targets))
  Lamda = np.zeros((len_class,np.sum(np.asarray(nb_class)),1))

  start = time.time()
  print ('Starting ...')
  th = 0
  indx = 0
  for i in range(len(inputs_train)):

    bic, log, para, Beta_hat, Sigma_hat, lambda_hat  = BIC_GPU(inputs_train[i])
    BIC_.append(bic)
    Log_lik[int(i/nb_iter_),i%nb_iter_,:] = np.asarray(log).reshape(iter_EM,)

    if (i + 1) % nb_iter_ == 0:
      Beta[th,indx:indx + nb_class[th],:,:] = Beta_hat
      if al_targ == True:
        Sigma[th,indx:indx + nb_class[th],:,:] = Sigma_hat
      else:
        Sigma[th,indx:indx + nb_class[th],:,:] = Sigma_hat.reshape(Sigma_hat.shape[0],1)
      Lamda[th,indx:indx + nb_class[th],:] = lambda_hat.reshape(lambda_hat.shape[0],1)
      indx = indx +  nb_class[th]
      th = th + 1

    if int(i/len(inputs_train)*100) % 5 == 0:
      print('Completing ...: ',int(i/len(inputs_train)*100),'%')

  print('Executive time: ',time.time() - start)
  # convert result to array and reshape it
  bic_arr = np.asarray(BIC_)
  bic_reshape = bic_arr.reshape(nb_iter_,len(nb_class),order='F')
  # save to .txt
  if save:
    np.savetxt('BIC_test_'+str(test)+'.txt', bic_reshape)
    np.savetxt('Log_test_'+str(test)+'.txt', Log_lik.reshape(len_class*nb_iter_,iter_EM))
    np.savetxt('Beta_test_'+str(test)+'.txt', Beta.reshape(len_class*np.sum(np.asarray(nb_class))*nb_features*n_targets))
    np.savetxt('Sigma_test_'+str(test)+'.txt', Sigma.reshape(len_class*np.sum(np.asarray(nb_class))*n_targets*n_targets))
    np.savetxt('Lamda_test_'+str(test)+'.txt', Lamda.reshape(len_class*np.sum(np.asarray(nb_class))*1))
