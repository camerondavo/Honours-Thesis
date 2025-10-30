import numpy as np
import matplotlib.pyplot as plt
from numpy import log
import ultranest
from ultranest.plot import cornerplot
from scipy.stats import poisson
import pandas as pd
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import astropy.units as u
from matplotlib import colors as mcolors
from ultranest.utils import resample_equal
from scipy.interpolate import interp1d
import scipy.stats as stats
import ultranest.stepsampler
import itertools
import gc
import json
import corner
import os

from my_functions import *

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
#Quadrupole prior
def quadrupole_prior(cube):
    
    params = cube.copy()
    params[0] = cube[0]*200 # 0 - 2000 (Nbar)
    params[1] = cube[1]/5 # 0 - 0.2 (Q)
    params[2] = cube[2]*2*np.pi # 0 - 2pi (l1)
    
    c1 = cube[3] # 0 - 1, used to calculate theta below. 
    params[3] = np.arccos(np.clip(1 - 2*c1, -1, 1)) #(b1)
    
    params[4] = cube[4]*2*np.pi # 0 - 2pi (l2)
    c2 = cube[5] # 0 - 1, used to calculate theta below.
    params[5] = np.arccos(np.clip(1 - 2*c2, -1, 1)) #(b2)
    
    return params

def vectorised_quadrupole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*200 # 0 - 200 (Nbar)
    params[:,1] = cube[:,1]/5 # 0 - 0.2 (Q)
    params[:,2] = cube[:,2]*2*np.pi # 0 - 2pi (l1)
    
    c1 = cube[:,3] # 0 - 1, used to calculate theta below. 
    params[:,3] = np.arccos(np.clip(1 - 2*c1, -1, 1)) #(b1)
    
    params[:,4] = cube[:,4]*2*np.pi # 0 - 2pi (l2)
    c2 = cube[:,5] 
    params[:,5] = np.arccos(np.clip(1 - 2*c2, -1, 1)) #(b2)
    
    return params
    
# Quadrupole likelihood
def quadrupole_likelihood(params):
    N_bar, Q, l1, b1, l2, b2 = params
    
    pixels = hp.pix2vec(NSIDE, np.arange(NPIX))
    
    a = hp.ang2vec(b1, l1)
    b = hp.ang2vec(b2, l2)
    
    Q_prime = np.outer(a, b)
    Q_star = 1/2 * (Q_prime + Q_prime.T)
    Q_hat = Q_star - np.trace(Q_star)/3

    f = Q*np.einsum('ij,i...,j...', Q_hat, pixels, pixels)
    
    lambda_i = N_bar * (1 + f)
    
    return np.sum(poisson.logpmf(m, mu=lambda_i))

def vectorised_quadrupole_likelihood(params):
    N_bar = params[:,0]
    Q = params[:,1]
    l1, b1 = params[:,2], params[:,3]
    l2, b2 = params[:,4], params[:,5]
    
    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))#.T  # Shape (NPIX, 3) ### (3, NPIX)

    v1 = hp.ang2vec(b1, l1)  # shape (n_samples, 3)
    v2 = hp.ang2vec(b2, l2)  # shape (n_samples, 3)

    Q_prime = np.einsum('ni,nj->nij', v1, v2)  # shape (n_samples, 3, 3)
    Q_star = 0.5 * (Q_prime + np.transpose(Q_prime, axes=(0, 2, 1)))  # shape (n_samples, 3, 3)
    trace = np.trace(Q_star, axis1=1, axis2=2)  # shape (n_samples,)
    Q_hat = Q_star - trace[:, None, None] / 3 * np.identity(3)  # shape (n_samples, 3, 3)
    
    f = Q * np.einsum('abc,b...,c...', Q_hat, pixels, pixels) # shape (NPIX, n_samples)
    
    lambda_i = N_bar * (1 + f) # shape (NPIX, n_samples)
    
    log_likelihoods = np.sum(poisson.logpmf(m, mu=lambda_i.T), axis=1)  # shape (n_samples,)
    return log_likelihoods

# read in each npy file from '../log_dir/testing/quadrupole/'
filepath = '../log_dir/testing/quadrupole/'
for file in os.listdir(filepath):
    if file.endswith('.npy'):
        m = np.load(filepath+file, allow_pickle=True)
        N = file.split('_')[2]
        
        #regular
        quad_param_names = [r'$\bar N$', 'Q', r'$\ell_1$', r'$b_1$', r'$\ell_2$', r'$b_2$']
        quadrupole_sampler = ultranest.ReactiveNestedSampler(quad_param_names, quadrupole_likelihood,
                                                                quadrupole_prior, log_dir=f'../log_dir/testing/quadrupole/comparisons/regular/nbar_{N}',
                                                                vectorized=False, resume=True)
        quadrupole_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                    nsteps=10,
                    generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))

        quadrupole_sampler = quadrupole_sampler.run()

        #Vectorised
        vectorised_quadrupole_sampler = ultranest.ReactiveNestedSampler(quad_param_names, vectorised_quadrupole_likelihood,
                                                                vectorised_quadrupole_prior, log_dir=f'../log_dir/testing/quadrupole/comparisons/vectorised/nbar_{N}',
                                                                vectorized=True, resume=True)

        vectorised_quadrupole_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                    nsteps=10,
                    generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))

        vectorised_quadrupole_sampler = vectorised_quadrupole_sampler.run()