import numpy as np
import matplotlib.pyplot as plt
from numpy import log
import ultranest
from ultranest.plot import cornerplot
from scipy.stats import poisson
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

# Monopole Prior
def vectorised_monopole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*220 
    
    return params

# Monopole Likelihood
def vectorised_monopole_likelihood(params):
    N_bar = params[:,0]    
    lambda_i = (N_bar[None, :] * np.ones((NPIX, 1)))
    return np.sum(poisson.logpmf(m, mu=lambda_i.T), axis=1)

# Dipole prior
def vectorised_dipole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*220 # 0 - 220
    params[:,1] = cube[:,1]/10 # 0 - 0.1
    params[:,2] = cube[:,2]*2*np.pi # 0 - 2pi
    
    c = cube[:,3] # 0 - 1, used to calculate theta below. 
    params[:,3] = np.arccos(np.clip(1 - 2*c, -1, 1)) #Clipping to ensure we stay in the valid range
    
    return params


# Dipole Likelihood
def vectorised_dipole_likelihood(params):
    N_bar = params[:,0]

    D = params[:,1]
    l, b = params[:,2], params[:,3]
    
    dipole_vec = hp.ang2vec(b, l)  # shape (n_samples, 3)
    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))  # Shape (3, NPIX)

    dot_product = np.dot(dipole_vec, pixels)  # shape (n_sample, NPIX)
    angles = np.arccos(dot_product) # shape (n_sample, NPIX)
    dipole_signal = D * np.cos(angles).T # shape (NPIX, n_samples), to match D shape

    lambda_i = N_bar * (1 + dipole_signal)  # shape (NPIX, n_samples)   
    log_likelihoods = np.sum(poisson.logpmf(m, mu=lambda_i.T), axis=1)  # shape (n_samples,)
    return log_likelihoods


# Quadrupole
def vectorised_quadrupole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*220 # 0 - 220 (Nbar)
    params[:,1] = cube[:,1]/5 # 0 - 0.2 (Q)
    params[:,2] = cube[:,2]*2*np.pi # 0 - 2pi (l1)
    
    c1 = cube[:,3] # 0 - 1, used to calculate theta below. 
    params[:,3] = np.arccos(np.clip(1 - 2*c1, -1, 1)) #(b1)
    
    params[:,4] = cube[:,4]*2*np.pi # 0 - 2pi (l2)
    c2 = cube[:,5] 
    params[:,5] = np.arccos(np.clip(1 - 2*c2, -1, 1)) #(b2)
    
    return params

    
def vectorised_quadrupole_likelihood(params):
    N_bar = params[:,0]
    Q = params[:,1]
    l1, b1 = params[:,2], params[:,3]
    l2, b2 = params[:,4], params[:,5]
    
    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))  # shape (3, NPIX)

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


# Dipole+quadrupole prior
def vectorised_dipole_quad_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*220 # 0 - 200 (Nbar)
    params[:,1] = cube[:,1]/10 # 0 - 0.1 (D)
    params[:,2] = cube[:,2]/5 # 0 - 0.2 (Q)
    
    params[:,3] = cube[:,3]*2*np.pi # 0 - 2pi (l)
    c = cube[:,4] # 0 - 1, used to calculate theta below. 
    params[:,4] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b)
    
    params[:,5] = cube[:,5]*2*np.pi # 0 - 2pi (l1)
    c = cube[:,6] # 0 - 1, used to calculate theta below. 
    params[:,6] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b1)
    
    params[:,7] = cube[:,7]*2*np.pi # 0 - 2pi (l2)
    c = cube[:,8] # 0 - 1, used to calculate theta below. 
    params[:,8] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b2)
    return params

# Dipole + Quadrupole Likelihood
def vectorised_dipole_quad_likelihood(params):
    # Extracting parameters from input
    N_bar = params[:, 0]
    D = params[:, 1]
    Q = params[:, 2]
    l, b = params[:, 3], params[:, 4]
    l1 , b1 = params[:, 5], params[:, 6]
    l2, b2 = params[:, 7], params[:, 8]

    # Converting the angles (l, b) into unit vectors (dipole_vec)
    dipole_vec = hp.ang2vec(b, l)  # shape (n_samples, 3)

    # Generating pixel vectors (pixels) from HEALPix
    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))  # Shape (3, NPIX)

    # angles = pixel_angles(pixels, dipole_vec) # shape (n_sample, NPIX)
    dot_product = np.dot(dipole_vec, pixels)  # shape (n_sample, NPIX)
    angles = np.arccos(dot_product) # shape (n_sample, NPIX)
    dipole_signal = D * np.cos(angles).T # shape (NPIX, n_samples), to mathc below
    
    # Converting l1, b1, l2, b2 into unit vectors for Q_prime calculation
    v1 = hp.ang2vec(b1, l1)  # shape (n_samples, 3)
    v2 = hp.ang2vec(b2, l2)  # shape (n_samples, 3)

    Q_prime = np.einsum('ni,nj->nij', v1, v2)  # shape (n_samples, 3, 3)
    Q_star = 0.5 * (Q_prime + np.transpose(Q_prime, axes=(0, 2, 1)))  # shape (n_samples, 3, 3)
    trace = np.trace(Q_star, axis1=1, axis2=2)  # shape (n_samples,)
    Q_hat = Q_star - trace[:, None, None] / 3 * np.identity(3)  # shape (n_samples, 3, 3)
    
    quad_signal = Q * np.einsum('abc,b...,c...', Q_hat, pixels, pixels) # shape (NPIX,n_samples)
    
    lambda_i = N_bar * (1 + dipole_signal + quad_signal) # shape (NPIX, n_samples)
    log_likelihoods = np.sum(poisson.logpmf(m, mu=lambda_i.T), axis=1)  # shape (n_samples,)

    return log_likelihoods

pathname = '../log_dir/Dipole_and_Quadrupole_Data/Changing_Amplitude/'
D = 0.005
Q_values = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015] #relative amps of D/Q: inf, 10, 5, 3.33, 2.5, 2, 1, 0.667, 0.5, 0.4, 0.333
Nbar_values = [4.1, 10.2, 20.3, 40.7, 61.0, 81.4, 101.7, 152.6, 203.5] # counts of 200,000, 500,000, 1,000,000, 2,000,000, 3,000,000, 4,000,000, 5,000,000, 7,5000,000, 10,000,000

mono_param_names = ['Nbar']
dipole_param_names = ['Nbar', 'D', 'l', 'b']
quadrupole_param_names = ['Nbar', 'Q', 'l1', 'b1', 'l2', 'b2']
dipole_quad_param_names = ['N', 'D', 'Q', 'l', 'b', 'l1', 'b1', 'l2', 'b2']

progress_file = pathname + 'progress.json'
# Read progress or initialize
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    start_Nbar_idx = progress['Nbar_idx']
    start_Q_idx = progress['Q_idx']
    start_iter = progress['Iteration']

else:
    start_Nbar_idx = 4
    start_Q_idx = 0
    start_iter = 0
for iteration in range(start_iter, 1):
    for Nbar_idx in range(start_Nbar_idx, len(Nbar_values)):
        N_bar = Nbar_values[Nbar_idx]
        for Q_idx in range(start_Q_idx, len(Q_values)):
            Q = Q_values[Q_idx]
            np.random.seed(42+iteration)
            data = np.load(pathname+f'Datasets/skymap_data_Nbar_{N_bar}_Q_{Q}_iteration_{iteration}.npy',
                        allow_pickle=True).item()
            m = data['m']
            metadata = data['metadata']
            NSIDE = metadata['NSIDE']
            NPIX = metadata['NPIX']

            # Monopole
            mono_sampler = ultranest.ReactiveNestedSampler(mono_param_names, vectorised_monopole_likelihood, vectorised_monopole_prior,
                log_dir=pathname + f'Monopole_model/Nbar_{N_bar}/Nbar_{N_bar}_Q_{Q}_iteration_{iteration}', vectorized=True, resume='resume-similar')
            
            mono_result = mono_sampler.run()

            # Dipole
            dipole_sampler = ultranest.ReactiveNestedSampler(dipole_param_names, vectorised_dipole_likelihood, vectorised_dipole_prior,
                log_dir=pathname + f'Dipole_model/Nbar_{N_bar}/Nbar_{N_bar}_Q_{Q}_iteration_{iteration}', vectorized=True, resume='resume-similar')
            dipole_result = dipole_sampler.run()

            # Quadrupole 
            quad_sampler = ultranest.ReactiveNestedSampler(quadrupole_param_names, vectorised_quadrupole_likelihood, vectorised_quadrupole_prior,
                log_dir=pathname + f'Quadrupole_model/Nbar_{N_bar}/Nbar_{N_bar}_Q_{Q}_iteration_{iteration}', vectorized=True, resume='resume-similar')

            quad_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=10, generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))
            quad_result = quad_sampler.run(max_ncalls=500000)

            # Dipole + Quadrupole 
            dq_sampler = ultranest.ReactiveNestedSampler(dipole_quad_param_names, vectorised_dipole_quad_likelihood, vectorised_dipole_quad_prior,
                log_dir=pathname + f'Dipole_Quadrupole_model/Nbar_{N_bar}/Nbar_{N_bar}_Q_{Q}_iteration_{iteration}', vectorized=True, resume='resume-similar')

            dq_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=10, generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))
            
            dq_result = dq_sampler.run(max_ncalls=500000) # needing to cap the number of calls to avoid memory issues I had on the cluster
            
            # Save progress after each mask angle
            with open(progress_file, 'w') as f:
                json.dump({'Nbar_idx': Nbar_idx, 'Q_idx': Q_idx+1, 'Iteration': iteration}, f)

        with open(progress_file, 'w') as f:
            json.dump({'Nbar_idx': Nbar_idx+1, 'Q_idx': 0, 'Iteration': iteration}, f)

    with open(progress_file, 'w') as f:
        json.dump({'Nbar_idx': 0, 'Q_idx': 0, 'Iteration': iteration+1}, f)

