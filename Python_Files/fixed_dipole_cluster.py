import numpy as np
import matplotlib.pyplot as plt
from numpy import log
import ultranest
from ultranest.plot import cornerplot

import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import astropy.units as u
from matplotlib import colors as mcolors
from ultranest.utils import resample_equal
from scipy.interpolate import interp1d
import scipy.stats as stats
import ultranest.stepsampler
import itertools
from mpi4py import MPI
import gc
import time
import json

from my_functions import *



# Priors and likelihood functions
# Monopole
def monopole_prior(cube):
    params = cube.copy()
    params[0] = cube[0]*100 # 0 - 200 (Nbar)
    
    return params

def monopole_likelihood(params):
    N_bar = params[0]
    
    lambda_i = N_bar * np.ones(NPIX)
    
    return np.sum(poisson.logpmf(m, mu=lambda_i))

# Dipole
def vectorised_dipole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*100 # 0 - 100

    # Set the bounds for a delta function around the fixed values
    eps_D = 1e-4
    eps = 1e-2
    D_bounds = (0.007-eps_D, 0.007+eps_D)
    l_bounds = (2*np.pi - np.radians(96)-eps, 2*np.pi - np.radians(96)+eps)
    b_bounds = (np.radians(48)-eps, np.radians(48)+eps)
    

    params[:,1] = cube[:, 1] * (D_bounds[1] - D_bounds[0]) + D_bounds[0]
    params[:,2] =  cube[:, 2] * (l_bounds[1] - l_bounds[0]) + l_bounds[0]
    params[:,3] = cube[:, 3] * (b_bounds[1] - b_bounds[0]) + b_bounds[0]

    return params
    
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
    params[:,0] = cube[:,0]*100 # 0 - 200 (Nbar)
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

# Dipole + Quadrupole
def vectorised_dipole_quad_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*120 # 0 - 100 (Nbar)

    eps_D = 1e-4
    eps = 1e-2
    D_bounds = (0.007-eps_D, 0.007+eps_D)
    l_bounds = (2*np.pi - np.radians(96)-eps, 2*np.pi - np.radians(96)+eps)
    b_bounds = (np.radians(48)-eps, np.radians(48)+eps)

    params[:,1] = cube[:, 1] * (D_bounds[1] - D_bounds[0]) + D_bounds[0]
    params[:,2] = cube[:,2]/5 # 0 - 0.2 (Q)
    
    params[:,3] =  cube[:, 3] * (l_bounds[1] - l_bounds[0]) + l_bounds[0]
    params[:,4] = cube[:, 4] * (b_bounds[1] - b_bounds[0]) + b_bounds[0]

    params[:,5] = cube[:,5]*2*np.pi # 0 - 2pi (l1)
    c = cube[:,6] # 0 - 1, used to calculate theta below. 
    params[:,6] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b1)
    
    params[:,7] = cube[:,7]*2*np.pi # 0 - 2pi (l2)
    c = cube[:,8] # 0 - 1, used to calculate theta below. 
    params[:,8] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b2)
    return params

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


def fit_models(N_bar, i, pathname):
# perform model 0 fitting (monopole)
    start = time.time()
    mono_sampler = ultranest.ReactiveNestedSampler(mono_param_names, monopole_likelihood, monopole_prior,
                    log_dir=pathname+f'monopole_model/monopole_Nbar_{N_bar}_{i}',
                                                    resume='resume-similar')
    mono_result = mono_sampler.run()
    end = time.time()
    #save the time taken to fit the model   
    with open(pathname+f'monopole_model/times/monopole_Nbar_{N_bar}_{i}_time.txt', 'w') as f:
        f.write(f'{end - start}')

#perform model 1 fitting (dipole)
    start = time.time()
    dipole_sampler = ultranest.ReactiveNestedSampler(dipole_param_names, vectorised_dipole_likelihood, vectorised_dipole_prior,
                    log_dir=pathname+f'dipole_model/dipole_Nbar_{N_bar}_{i}', vectorized=True,
                                                    resume='resume-similar')
    dipole_result = dipole_sampler.run()
    end = time.time()

    #save the time taken to fit the model
    with open(pathname+f'dipole_model/times/dipole_Nbar_{N_bar}_{i}_time.txt', 'w') as f:
        f.write(f'{end - start}')
    
#perform model 2 fitting (quadrupole)
    start = time.time()
    quad_sampler = ultranest.ReactiveNestedSampler(quad_param_names, vectorised_quadrupole_likelihood, vectorised_quadrupole_prior, 
        log_dir=pathname+f'quadrupole_model/quadrupole_Nbar_{N_bar}_{i}', vectorized=True,
                                                    resume='resume-similar')

    quad_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=10,
        generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))

    quad_result = quad_sampler.run()
    end = time.time()

    #save the time taken to fit the model
    with open(pathname+f'quadrupole_model/times/quadrupole_Nbar_{N_bar}_{i}_time.txt', 'w') as f:
        f.write(f'{end - start}')
    
#perform model 3 fitting (dipole and quadrupole)
    start = time.time()
    dipole_quad_sampler = ultranest.ReactiveNestedSampler(dipole_quad_param_names, vectorised_dipole_quad_likelihood, 
                                                          vectorised_dipole_quad_prior,
            log_dir=pathname+f'dipole_quadrupole_model/dipole_quadrupole_Nbar_{N_bar}_{i}',
                                                    vectorized=True, resume='resume-similar')

    dipole_quad_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=10,
        generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))

    dipole_quad_result = dipole_quad_sampler.run()
    end = time.time()

    #save the time taken to fit the model
    with open(pathname+f'dipole_quadrupole_model/times/dipole_quadrupole_Nbar_{N_bar}_{i}_time.txt', 'w') as f:
        f.write(f'{end - start}')    

    #perform model 4 fitting (to come)
    
    # print(f'Finished: Nbar_{N_bar_true}_D_{D_true}_Q_{Q_true} \n')
    
    return


pathname='./log_dir/Dipole_and_Quadrupole_Data/Fixed_dipole/'

mono_param_names = [r'$\bar N$']
dipole_param_names = [r'$\bar N$', 'D', r'$\ell$', 'b']
quad_param_names = [r'$\bar N$', 'Q', r'$\ell_1$', r'$b_1$', r'$\ell_2$', r'$b_2$']
dipole_quad_param_names = [r'$\bar N$', 'D', 'Q', r'$\ell$', r'$b$', r'$\ell_1$', r'$b_1$', r'$\ell_2$', r'$b_2$']

N_bars = [1,5,10,20,40,60,80,100]

# Checking progress and starting from the last completed Nbar and i
progress_file = 'progress.json'
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    start_Nbar_idx = progress['Nbar_idx']
    start_i = progress['i']
else:
    start_Nbar_idx = 0
    start_i = 0

for Nbar_idx in range(start_Nbar_idx, len(N_bars)):
    N_bar = N_bars[Nbar_idx]

    for i in range(start_i, 500):
        # First load the data and metadata
        data = np.load(pathname+f'Datasets/Raw_Files/Nbar_{N_bar}/skymap_data_Nbar_{N_bar}_{i}.npy',
                        allow_pickle=True).item()
        m = data['m']
        metadata = data['metadata']
        NSIDE = metadata['NSIDE']
        NPIX = metadata['NPIX']
        
        fit_models(N_bar, i, pathname)
        del data, m, metadata
        gc.collect()

        # Save progress after each iteration
        with open(progress_file, 'w') as f:
            json.dump({'Nbar_idx': Nbar_idx, 'i': i+1}, f)
    
    # Reset start_i after finishing inner loop
    start_i = 0

