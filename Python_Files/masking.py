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
import time
import csv

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
    lambda_i_masked = lambda_i[fit_mask]
    return np.sum(poisson.logpmf(m_fit, mu=lambda_i_masked.T), axis=1)

# Dipole prior
def vectorised_dipole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*220 # 0 - 220
    params[:,1] = cube[:,1]/10 # 0 - 0.1
    params[:,2] = cube[:,2]*2*np.pi # 0 - 2pi
    
    c = cube[:,3] # 0 - 1, used to calculate theta below. 
    params[:,3] = np.arccos(np.clip(1 - 2*c, -1, 1)) #Clipping to ensure we stay in the valid range
    
    return params

# Fixed Dipole
def vectorised_fixed_dipole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*220 # 0 - 220

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
    lambda_i_masked = lambda_i[fit_mask]
    log_likelihoods = np.sum(poisson.logpmf(m_fit, mu=lambda_i_masked.T), axis=1)  # shape (n_samples,)
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
    lambda_i_masked = lambda_i[fit_mask]
    log_likelihoods = np.sum(poisson.logpmf(m_fit, mu=lambda_i_masked.T), axis=1)  # shape (n_samples,)
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
    lambda_i_masked = lambda_i[fit_mask]
    log_likelihoods = np.sum(poisson.logpmf(m_fit, mu=lambda_i_masked.T), axis=1)  # shape (n_samples,)

    return log_likelihoods

# Masking function
def mask_data(NSIDE, m, mask_angle):
    NPIX = hp.nside2npix(NSIDE)
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (theta > np.radians(90 - mask_angle)) & (theta < np.radians(90+mask_angle))
    
    # for plotting, need to use nans
    m_plot = m.copy()
    m_plot = m_plot.astype(float)  # convert to float so np.nan is allowed
    m_plot[mask] = np.nan

    # for fitting, can't use nans need to remove the masked indices. Also need fit_mask to mask in likelihood too
    m_fit = m.copy()
    fit_mask = ~mask
    m_fit = m[fit_mask]

    return m_plot, m_fit, fit_mask

# Dipole sampling
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
D = 0.007
Q = 0.014 
dipole_theta = np.deg2rad(48) # CMB Directions
dipole_phi = -np.deg2rad(360-264)
# Set the quadrupole vectors to both point at the CMB dipole direction
q_vector_1 = [dipole_theta, dipole_phi]  # l1, b1
q_vector_2 = [dipole_theta, dipole_phi]  # l2, b2
# np.random.seed(42) 
# m, lambda_ = dipole_sampling(NSIDE, N_bar=N_bar, D=D, dipole_theta=dipole_theta, dipole_phi=dipole_phi)
mono_param_names = ['Nbar']
dipole_param_names = ['Nbar', 'D', 'l', 'b']
quadrupole_param_names = ['Nbar', 'Q', 'l1', 'b1', 'l2', 'b2']
dipole_quad_param_names = ['N', 'D', 'Q', 'l', 'b', 'l1', 'b1', 'l2', 'b2']

# Apply masking, will turn this into a function soon
mask_angles = [0,10,20,30,40,50,60,70,80]
N_bars = [4.1, 10.2, 20.3, 40.7, 101.7, 203.5] # counts of 200,000, 500,000, 1,000,000, 2,000,000, 5,000,000, 10,000,000

# data = 'dipole'
data = 'dipole_quad' 

if data == 'dipole':
    progress_file = 'dipole_masked_progress.json'
    pathname = '../log_dir/Dipole_Data/Masking/'
    N_bars = [1.0, 2.0, 4.1, 10.2, 20.3, 40.7, 101.7, 203.5] # counts of 50,000, 100,000, 200,000, 500,000, 1,000,000, 2,000,000, 5,000,000, 10,000,000

    # Read progress or initialize
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        start_Nbar_idx = progress['Nbar_idx']
        start_i = progress['i']
        start_mask_idx = progress['mask_idx']
    else:
        start_Nbar_idx = 0
        start_i = 0
        start_mask_idx = 0

    for Nbar_idx in range(start_Nbar_idx, len(N_bars)):
        N_bar = N_bars[Nbar_idx]

        for i in range(start_i, 20): # just run iteration 4
            # Generate and save the data, seeded for reproducibility
            np.random.seed(42+i)
            m, lambda_ = dipole_sampling(NSIDE, N_bar=N_bar, D=D,
                                            dipole_theta=dipole_theta,
                                            dipole_phi=dipole_phi)

            metadata = {
                'Nbar': N_bar,
                'D': D,
                'NSIDE': NSIDE,
                'NPIX': NPIX,
                'dipole_theta': dipole_theta,
                'dipole_phi': dipole_phi,
            }

            data_to_save = {'m': m, 'metadata': metadata}
            np.save(pathname + f'Datasets/skymap_data_Nbar_{N_bar}_{i}.npy', data_to_save)

            # For each mask angle, mask the data and run the samplers
            for mask_idx in range(start_mask_idx, len(mask_angles)):
                mask_angle = mask_angles[mask_idx]
                m_plot, m_fit, fit_mask = mask_data(NSIDE, m, mask_angle)

                # Monopole
                mono_sampler = ultranest.ReactiveNestedSampler(mono_param_names, vectorised_monopole_likelihood, vectorised_monopole_prior,
                    log_dir=pathname + f'Monopole_model/Nbar_{N_bar}/Nbar_{N_bar}_mask_{mask_angle}_iteration_{i}', vectorized=True, resume=True)
                
                mono_result = mono_sampler.run()

                # Dipole
                dipole_sampler = ultranest.ReactiveNestedSampler(dipole_param_names, vectorised_dipole_likelihood, vectorised_dipole_prior,
                    log_dir=pathname + f'Dipole_model/Nbar_{N_bar}/Nbar_{N_bar}_mask_{mask_angle}_iteration_{i}', vectorized=True, resume=True)
                dipole_result = dipole_sampler.run()

                # Save progress after each mask angle
                with open(progress_file, 'w') as f:
                    json.dump({'Nbar_idx': Nbar_idx, 'i': i, 'mask_idx': mask_idx + 1}, f)

            # Reset mask index after finishing all mask_angles
            start_mask_idx = 0

            # Save progress after finishing all masks for a given i
            with open(progress_file, 'w') as f:
                json.dump({'Nbar_idx': Nbar_idx, 'i': i + 1, 'mask_idx': 0}, f)

        # Reset i after N_bar completed
        start_i = 0


elif data == 'dipole_quad':
    progress_file = 'dq_masked_progress.json'
    pathname = '../log_dir/Dipole_and_Quadrupole_Data/Masking/'
    N_bars = [4.1, 10.2, 20.3, 40.7, 101.7, 203.5] # counts of 200,000, 500,000, 1,000,000, 2,000,000, 5,000,000, 10,000,000

    # Read progress or initialize
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        start_Nbar_idx = progress['Nbar_idx']
        start_i = progress['i']
        start_mask_idx = progress['mask_idx']
    else:
        start_Nbar_idx = 0
        start_i = 0
        start_mask_idx = 0

    # for i in range(start_i, 20):
    for i in range(0, 20): # doing nbar=4.1 only it was skipped
        # for Nbar_idx in range(start_Nbar_idx, len(N_bars)):
        for Nbar_idx in range(0, 1):
            N_bar = N_bars[Nbar_idx]

            # Generate and save the data, seeded for reproducibility
            np.random.seed(42+i)
            m, lambda_ = dipole_quad_sampling(NSIDE, N_bar=N_bar, D=D, Q=Q,
                                            d_vector=[dipole_theta, dipole_phi],
                                            q_vector_1=q_vector_1, q_vector_2=q_vector_2)

            metadata = {
                'Nbar': N_bar,
                'D': D,
                'Q': Q,
                'NSIDE': NSIDE,
                'NPIX': NPIX,
                'd_vector': [dipole_theta, dipole_phi],
                'q_vector_1': q_vector_1,
                'q_vector_2': q_vector_2,
            }

            data_to_save = {'m': m, 'metadata': metadata}
            np.save(pathname + f'Datasets/skymap_data_Nbar_{N_bar}_{i}.npy', data_to_save)

            # For each mask angle, mask the data and run the samplers
            for mask_idx in range(start_mask_idx, len(mask_angles)):
                mask_angle = mask_angles[mask_idx]
                m_plot, m_fit, fit_mask = mask_data(NSIDE, m, mask_angle)

                # Monopole
                start = time.time()
                mono_sampler = ultranest.ReactiveNestedSampler(mono_param_names, vectorised_monopole_likelihood, vectorised_monopole_prior,
                    log_dir=pathname + f'Monopole_model/Nbar_{N_bar}/Nbar_{N_bar}_mask_{mask_angle}_iteration_{i}', vectorized=True, resume=True)
                
                mono_result = mono_sampler.run()
                end = time.time()
                mono_time = end - start

                # Dipole
                start = time.time()
                dipole_sampler = ultranest.ReactiveNestedSampler(dipole_param_names, vectorised_dipole_likelihood, vectorised_dipole_prior,
                    log_dir=pathname + f'Dipole_model/Nbar_{N_bar}/Nbar_{N_bar}_mask_{mask_angle}_iteration_{i}', vectorized=True, resume=True)
                dipole_result = dipole_sampler.run()
                end = time.time()
                dipole_time = end - start

                # Quadrupole 
                start = time.time()
                quad_sampler = ultranest.ReactiveNestedSampler(quadrupole_param_names, vectorised_quadrupole_likelihood, vectorised_quadrupole_prior,
                    log_dir=pathname + f'Quadrupole_model/Nbar_{N_bar}/Nbar_{N_bar}_mask_{mask_angle}_iteration_{i}', vectorized=True, resume=True)
                
                quad_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                    nsteps=10, generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))
                quad_result = quad_sampler.run()
                end = time.time()
                quad_time = end - start

                # Dipole + Quadrupole 
                start = time.time()
                dq_sampler = ultranest.ReactiveNestedSampler(dipole_quad_param_names, vectorised_dipole_quad_likelihood, vectorised_dipole_quad_prior,
                    log_dir=pathname + f'Dipole_Quadrupole_model/Nbar_{N_bar}/Nbar_{N_bar}_mask_{mask_angle}_iteration_{i}', vectorized=True, resume=True)
                
                dq_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                    nsteps=10, generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))
                
                dq_result = dq_sampler.run(max_ncalls=1000000) # needing to cap the number of calls to avoid memory issues I had on the cluster
                end = time.time()
                dq_time = end - start

                # data to save
                times_data = {
                    'Nbar': N_bar,
                    'iteration': i,
                    'mask_angle': mask_angle,
                    'mono_time': mono_time,
                    'dipole_time': dipole_time,
                    'quad_time': quad_time,
                    'dq_time': dq_time
                }

                # choose one file to hold all results
                filename = pathname + "Times/all_times.csv"

                # check if file already exists
                file_exists = os.path.isfile(filename)

                # open in append mode
                with open(filename, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=times_data.keys())
                    if not file_exists:
                        writer.writeheader()   # only write header once
                    writer.writerow(times_data)
                    
                # Save progress after each mask angle
                with open(progress_file, 'w') as f:
                    json.dump({'Nbar_idx': Nbar_idx, 'i': i, 'mask_idx': mask_idx + 1}, f)

            # Reset mask index after finishing all mask_angles
            start_mask_idx = 0

            # Save progress after finishing all masks for a given i
            with open(progress_file, 'w') as f:
                json.dump({'Nbar_idx': Nbar_idx+1, 'i': i, 'mask_idx': 0}, f)
            
                # Save progress after finishing all masks for a given i
        with open(progress_file, 'w') as f:
            json.dump({'Nbar_idx': 0, 'i': i+1, 'mask_idx': 0}, f)





