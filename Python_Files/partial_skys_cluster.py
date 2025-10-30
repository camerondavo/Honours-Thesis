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

def draw_vectors(n):
    """Draw n random vectors on the sphere."""
    c = np.random.uniform(0,1,n)
    theta = np.arccos(1 - 2 * c)
    phi = np.random.uniform(0, 2 * np.pi, n)

    vectors = np.array([theta, phi]).T
    return vectors

def partial_sky_masking(m, theta, phi, r=40): 
    # smooth_map = compute_smooth_map(m, angle_scale=1)
    vec = hp.ang2vec(theta, phi)
    ipix_disc = hp.query_disc(nside=NSIDE, vec=vec, radius=np.radians(r))

    mask = np.zeros(NPIX, dtype=bool)
    mask[ipix_disc] = True  # Set the disc pixels to True

    # For plotting, we need the array to be of length NPIX
    m_plot = hp.ma(m.copy().astype(float))
    m_plot.mask = ~mask

    # m_plot_smooth = hp.ma(smooth_map.copy().astype(float))
    # m_plot_smooth.mask = ~mask

    # for fitting, can't use nans need to remove the masked indices. Also need fit_mask to mask in likelihood too
    m_fit = m.copy()
    fit_mask = mask
    m_fit = m[fit_mask]

    return m_plot, m_fit, fit_mask # removed smooth_map and m_plot_smooth bc I'm not plotting it here

# PRIORS
def vectorised_monopole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0] * 200 + 900
    
    return params

def vectorised_dipole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0] * 200 + 900 # 900 - 1100 (Nbar)
    params[:,1] = cube[:,1]/10 # 0 - 0.1
    params[:,2] = cube[:,2]*2*np.pi # 0 - 2pi
    
    c = cube[:,3] # 0 - 1, used to calculate theta below. 
    params[:,3] = np.arccos(np.clip(1 - 2*c, -1, 1)) #Clipping to ensure we stay in the valid range
    
    return params

def vectorised_quadrupole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0] * 200 + 900 # 900 - 1100 (Nbar)
    params[:,1] = cube[:,1]/5 # 0 - 0.2 (Q)
    params[:,2] = cube[:,2]*2*np.pi # 0 - 2pi (l1)
    
    c1 = cube[:,3] # 0 - 1, used to calculate theta below. 
    params[:,3] = np.arccos(np.clip(1 - 2*c1, -1, 1)) #(b1)
    
    params[:,4] = cube[:,4]*2*np.pi # 0 - 2pi (l2)
    c2 = cube[:,5] 
    params[:,5] = np.arccos(np.clip(1 - 2*c2, -1, 1)) #(b2)
    
    return params

def vectorised_dipole_quad_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0] * 200 + 900 # 900 - 1100 (Nbar)
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

# LIKELIHOODS
def vectorised_monopole_likelihood(params):
    N_bar = params[:,0]    
    lambda_i = (N_bar[None, :] * np.ones((NPIX, 1)))
    if masking == True:
        lambda_i = lambda_i[fit_mask]
    return np.sum(poisson.logpmf(m_fit, mu=lambda_i.T), axis=1)

def vectorised_dipole_likelihood(params):
    N_bar = params[:,0]
    D = params[:,1]
    l, b = params[:,2], params[:,3]

    # pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))  # Shape (3, NPIX)
    vec = hp.pix2vec(NSIDE, np.arange(NPIX))
    vectors = [vec[0], vec[1], vec[2]]  # shape (3, NPIX)
    pixels = [vectors[0][fit_mask],vectors[1][fit_mask],vectors[2][fit_mask]] 

    dipole_vec = hp.ang2vec(b, l)  # shape (n_samples, 3)
    dot_product = np.dot(dipole_vec, pixels)  # shape (n_sample, NPIX)
    angles = np.arccos(dot_product) # shape (n_samples, NPIX)
    dipole_signal = D * np.cos(angles).T # shape (N

    lambda_i = N_bar * (1 + dipole_signal)  # shape (NPIX, n_samples)    

    log_likelihoods = np.sum(poisson.logpmf(m_fit, mu=lambda_i.T), axis=1)  # shape (n_samples,)
    return log_likelihoods

def vectorised_quadrupole_likelihood(params):
    N_bar = params[:,0]
    Q = params[:,1]
    l1, b1 = params[:,2], params[:,3]
    l2, b2 = params[:,4], params[:,5]
    
    vec = hp.pix2vec(NSIDE, np.arange(NPIX))
    vectors = [vec[0], vec[1], vec[2]]  # shape (3, NPIX)
    pixels = [vectors[0][fit_mask],vectors[1][fit_mask],vectors[2][fit_mask]]  # Remove masked pixels
    # pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))  # shape (3, NPIX)
    
    v1 = hp.ang2vec(b1, l1)  # shape (n_samples, 3)
    v2 = hp.ang2vec(b2, l2)  # shape (n_samples, 3)

    Q_prime = np.einsum('ni,nj->nij', v1, v2)  # shape (n_samples, 3, 3)
    Q_star = 0.5 * (Q_prime + np.transpose(Q_prime, axes=(0, 2, 1)))  # shape (n_samples, 3, 3)
    trace = np.trace(Q_star, axis1=1, axis2=2)  # shape (n_samples,)
    Q_hat = Q_star - trace[:, None, None] / 3 * np.identity(3)  # shape (n_samples, 3, 3)
    
    f = Q * np.einsum('abc,b...,c...', Q_hat, pixels, pixels) # shape (NPIX, n_samples)
    
    lambda_i = N_bar * (1 + f) # shape (NPIX, n_samples)

    log_likelihoods = np.sum(poisson.logpmf(m_fit, mu=lambda_i.T), axis=1)  # shape (n_samples,)
    return log_likelihoods

def vectorised_dipole_quad_likelihood(params):
    N_bar = params[:, 0]
    D = params[:, 1]
    Q = params[:, 2]
    l, b = params[:, 3], params[:, 4]
    l1 , b1 = params[:, 5], params[:, 6]
    l2, b2 = params[:, 7], params[:, 8]
    
    # pixels, masked
    vec = hp.pix2vec(NSIDE, np.arange(NPIX))
    vectors = [vec[0], vec[1], vec[2]]  # shape (3, NPIX)
    pixels = [vectors[0][fit_mask],vectors[1][fit_mask],vectors[2][fit_mask]]  # Remove masked pixels
    # pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))  # shape (3, NPIX)

    # dipole signal
    dipole_vec = hp.ang2vec(b, l)  # shape (n_samples, 3)
    dot_product = np.dot(dipole_vec, pixels)  # shape (n_sample, NPIX)
    angles = np.arccos(dot_product) # shape (n_samples, NPIX)
    dipole_signal = D * np.cos(angles).T # shape (N

    # Quadrupole signal
    v1 = hp.ang2vec(b1, l1)  # shape (n_samples, 3)
    v2 = hp.ang2vec(b2, l2)  # shape (n_samples, 3)

    Q_prime = np.einsum('ni,nj->nij', v1, v2)  # shape (n_samples, 3, 3)
    Q_star = 0.5 * (Q_prime + np.transpose(Q_prime, axes=(0, 2, 1)))  # shape (n_samples, 3, 3)
    trace = np.trace(Q_star, axis1=1, axis2=2)  # shape (n_samples,)
    Q_hat = Q_star - trace[:, None, None] / 3 * np.identity(3)  # shape (n_samples, 3, 3)
    
    quad_signal = Q * np.einsum('abc,b...,c...', Q_hat, pixels, pixels) # shape (NPIX, n_samples)
    
    lambda_i = N_bar * (1 + dipole_signal+ quad_signal) # shape (NPIX, n_samples)

    log_likelihoods = np.sum(poisson.logpmf(m_fit, mu=lambda_i.T), axis=1)  # shape (n_samples,)
    return log_likelihoods


NSIDE_centres = 3
NPIX_centres = hp.nside2npix(NSIDE_centres)

NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
dipole_theta = np.deg2rad(48) # CMB Directions
dipole_phi = -np.deg2rad(360-264)

# vectors = draw_vectors(2)
q_vector_1 = [dipole_theta, dipole_phi]  # l1, b1
q_vector_2 = [dipole_theta, dipole_phi]  # l2, b2

mono_param_names = ['Nbar']
dipole_param_names = ['Nbar', 'D', 'l', 'b']
quadrupole_param_names = ['Nbar', 'Q', 'l1', 'b1', 'l2', 'b2']
dipole_quad_param_names = ['N', 'D', 'Q', 'l', 'b', 'l1', 'b1', 'l2', 'b2']

progress_file = 'partial_skies_progress.json'

N_bar = 1000
D = 0.007
Q = 0.014
masking = True  # Set to True if you want to apply the masking
model = 'Dipole'
# model = 'Dipole_and_Quadrupole'  # Change to 'Dipole_Quadrupole' if you want to run that model
pathname = f'../log_dir/{model}_Data/Partial_Skies/'


    # Read progress or initialize
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    start_i = progress['Iteration']
    start_pix = progress['Pixel']
else:
    start_i = 0
    start_pix = 0

for i in range(start_i, 20):
    # Generate the data based on the chosen model
    if model == 'Dipole':
        
        # Generate and save the skymap
        np.random.seed(42)  # For reproducibility
        m, lambda_i = dipole_sampling(NSIDE, N_bar=N_bar, D=D, dipole_theta=dipole_theta, dipole_phi=dipole_phi)

        metadata = {
                    'Nbar': N_bar,
                    'D': D,
                    'NSIDE': NSIDE,
                    'NPIX': NPIX,
                    'dipole_theta': dipole_theta,
                    'dipole_phi': dipole_phi,
                }

        data_to_save = {'m': m, 'metadata': metadata}
        np.save(pathname + f'Datasets/skymap_data_Nbar_{N_bar}_iteration_{i}.npy', data_to_save)

    elif model =='Dipole_and_Quadrupole':
        # Generarte and save the skymap
        np.random.seed(42)  # For reproducibility
        m, lambda_i = dipole_quad_sampling(NSIDE, N_bar=N_bar, D=D, Q=Q, d_vector=[dipole_theta,dipole_phi], q_vector_1=q_vector_1, q_vector_2=q_vector_2)
            
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
        np.save(pathname + f'Datasets/skymap_data_Nbar_{N_bar}_iteration_{i}.npy', data_to_save)


    # Loop through, masking at each of the pixels in 
    for pixel in range(start_i, NPIX_centres):
        theta, phi = hp.pix2ang(NSIDE_centres, pixel)
        # phi_plot = (phi + np.pi) % (2 * np.pi) - np.pi # only need this is we are trying to plot

        # Mask the skymap
        m_plot, m_fit, fit_mask = partial_sky_masking(m, theta, phi, r=40)
        
        # Run the nested sampling for each model
        # Monopole
        mono_sampler = ultranest.ReactiveNestedSampler(mono_param_names, vectorised_monopole_likelihood, vectorised_monopole_prior,
            log_dir=pathname + f'Monopole_model/Nbar_{N_bar}_pixel_{pixel}_iteration_{i}', vectorized=True, resume=True)

        mono_result = mono_sampler.run()

        # Dipole
        dipole_sampler = ultranest.ReactiveNestedSampler(dipole_param_names, vectorised_dipole_likelihood, vectorised_dipole_prior,
            log_dir=pathname + f'Dipole_model/Nbar_{N_bar}_pixel_{pixel}_iteration_{i}', vectorized=True, resume=True)
        dipole_result = dipole_sampler.run()

        if model == 'Dipole_and_Quadrupole':
            # Quadrupole 
            quad_sampler = ultranest.ReactiveNestedSampler(quadrupole_param_names, vectorised_quadrupole_likelihood, vectorised_quadrupole_prior,
                log_dir=pathname + f'Quadrupole_model/Nbar_{N_bar}_pixel_{pixel}_iteration_{i}', vectorized=True, resume=True)

            quad_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=10, generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))
            
            quad_result = quad_sampler.run()

            # Dipole + Quadrupole 
            dq_sampler = ultranest.ReactiveNestedSampler(dipole_quad_param_names, vectorised_dipole_quad_likelihood, vectorised_dipole_quad_prior,
                log_dir=pathname + f'Dipole_Quadrupole_model/Nbar_{N_bar}_pixel_{pixel}_iteration_{i}', vectorized=True, resume=True)

            dq_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
                nsteps=10, generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))
            
            dq_result = dq_sampler.run(max_ncalls=1000000) # needing to cap the number of calls to avoid memory issues I had on the cluster
            
        # move to next pixel for the iteration
        with open(progress_file, 'w') as f:
            json.dump({'Iteration': i, 'Pixel': pixel+1}, f)
        gc.collect()
        
    # Move to the next iteration and restart pixels
    with open(progress_file, 'w') as f:
        json.dump({'Iteration': i+1, 'Pixel': 0}, f)
        gc.collect()



