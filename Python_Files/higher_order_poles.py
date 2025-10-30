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
from mpi4py import MPI
import gc

from my_functions import *

# Define relevant models
# dipole signal
def dipole_sampling(NSIDE, N_bar, D, dipole_theta, dipole_phi):
    NPIX = hp.nside2npix(NSIDE)
    dipole_vec = hp.ang2vec(dipole_theta, dipole_phi) # (x, y, z) from (b, l)    
    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))
    
    dot_product = np.dot(dipole_vec, pixels)  
    angles = np.arccos(dot_product)
    dipole_signal = D*np.cos(angles)

    lambda_ = N_bar * (1 + dipole_signal)
    sample = np.random.poisson(lambda_)

    return sample, lambda_

def dipole_plotting(m, dipole_theta=None, dipole_phi=None, title=None, unit=None, cmap='viridis'):
    projview(m, title=title, unit=unit,
    graticule=True, graticule_labels=True, projection_type="mollweide", cmap=cmap);

    newprojplot(theta=dipole_theta, phi=dipole_phi, marker="*", color="k", markersize=15);
    # plt.tight_layout()
    return

# Dipole prior
def dipole_prior(cube):
    params = cube.copy()
    params[0] = cube[0]*200 # 0 - 100
    params[1] = cube[1]/10 # 0 - 0.1
    params[2] = cube[2]*2*np.pi # 0 - 2pi
    
    c = cube[3] # 0 - 1, used to calculate theta below. 
    params[3] = np.arccos(np.clip(1 - 2*c, -1, 1)) #Clipping to ensure we stay in the valid range
    
    return params

def vectorised_dipole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*200 # 0 - 100
    params[:,1] = cube[:,1]/10 # 0 - 0.1
    params[:,2] = cube[:,2]*2*np.pi # 0 - 2pi
    
    c = cube[:,3] # 0 - 1, used to calculate theta below. 
    params[:,3] = np.arccos(np.clip(1 - 2*c, -1, 1)) #Clipping to ensure we stay in the valid range
    
    return params

# Dipole likelihood
def dipole_likelihood(params):
    N_bar, D, l, b = params
    dipole_vec = hp.ang2vec(b, l) # dipole vector in cartesian
    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))
    
    dot_product = np.dot(dipole_vec, pixels)  
    angles = np.arccos(dot_product)
    dipole_signal = D*np.cos(angles)
    
    lambda_i = N_bar * (1 + dipole_signal)
    
    return np.sum(poisson.logpmf(m, mu=lambda_i))

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


#quadrupole signal
def quadrupole_sampling(NSIDE, N_bar, Q, vector_1, vector_2):
    NPIX = hp.nside2npix(NSIDE)
    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))
    b1, l1 = vector_1[0], vector_1[1] # vectors are (theta, phi) which is (b,l)
    b2, l2 = vector_2[0], vector_2[1] # vectors are (theta, phi) which is (b,l)
    
    # Constructing quadrupole tensor
    a = hp.ang2vec(b1, l1) #(theta, phi) to (x, y, z)
    b = hp.ang2vec(b2, l2) #(theta, phi) to (x, y, z)
    Q_prime = np.outer(a, b)
    Q_star = 1/2 * (Q_prime + Q_prime.T)
    Q_hat = Q_star - (np.trace(Q_star)/3 * np.eye(3))

    # Quadrupole signal
    quad_signal = Q*np.einsum('ij,i...,j...', Q_hat, pixels, pixels)
    
    lambda_ = N_bar * (1 + quad_signal)
    sample = np.random.poisson(lambda_)
    return sample, lambda_

def quadrupole_plotting(m, vector_1=None, vector_2=None, title=None, unit=None, cmap='viridis'):
    projview(m, title=title, unit=unit,
    graticule=True, graticule_labels=True, projection_type="mollweide", cmap=cmap);
    if vector_1[1] > np.pi:
        vector_1[1] = vector_1[1]- 2*np.pi
    if vector_2[1] > np.pi:
        vector_2[1] = vector_2[1]- 2*np.pi    
    newprojplot(theta=vector_1[0], phi=vector_1[1], marker="*", color="cyan", markersize=15);
    newprojplot(theta=vector_2[0], phi=vector_2[1], marker="*", color="cyan", markersize=15);
    plt.tight_layout()

    return

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
    
    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))
    
    v1 = hp.ang2vec(b1, l1)
    v2 = hp.ang2vec(b2, l2)
    
    Q_prime = np.outer(v1, v2)
    Q_star = 1/2 * (Q_prime + Q_prime.T)
    Q_hat = Q_star - (np.trace(Q_star)/3 * np.eye(3))

    quad_signal = Q*np.einsum('ij,i...,j...', Q_hat, pixels, pixels)
    
    lambda_i = N_bar * (1 + quad_signal)
    
    return np.sum(poisson.logpmf(m, mu=lambda_i))

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


#dipole+qudrupole signal
def dipole_quad_sampling(NSIDE, N_bar, D, Q, d_vector, q_vector_1, q_vector_2):
    NPIX = hp.nside2npix(NSIDE)
    dipole_theta, dipole_phi = d_vector[0], d_vector[1]
    dipole_vec = hp.ang2vec(dipole_theta, dipole_phi) # (x, y, z) from (b, l) shape (1, 3)

    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX))) # shape (3, NPIX)
    
    dot_product = np.dot(dipole_vec, pixels)  # returns shape (NPIX, )
    angles = np.arccos(dot_product) # shape (NPIX, )
    dipole_signal = D*np.cos(angles) # shape (NPIX, )
    
    # Quadrupole tensor
    b1, l1 = q_vector_1[0], q_vector_1[1]
    b2, l2 = q_vector_2[0], q_vector_2[1]
    
    # Constructing quadrupole tensor using the outer product of the 2 vectors
    v1 = hp.ang2vec(b1, l1)
    v2 = hp.ang2vec(b2, l2)
    Q_prime = np.outer(v1, v2)
    Q_star = 1/2 * (Q_prime + Q_prime.T)
    Q_hat = Q_star - (np.trace(Q_star)/3 * np.eye(3))

    # Quadrupole signal
    quad_signal = Q*np.einsum('ij,i...,j...', Q_hat, pixels, pixels)
    
    lambda_ = N_bar * (1 + dipole_signal + quad_signal)
    sample = np.random.poisson(lambda_)
    
    return sample, lambda_

def mixed_plotting(m, d_vector=None, q_vector_1=None, q_vector_2=None, title=None, unit=None, cmap='viridis'):
    projview(m, title=title, unit=unit,
    graticule=True, graticule_labels=True, projection_type="mollweide", cmap=cmap);
    
    newprojplot(theta=d_vector[0], phi=d_vector[1], marker="*", color="b", markersize=15);
    newprojplot(theta=q_vector_1[0], phi=q_vector_1[1], marker="*", color="lightskyblue", markersize=15);
    newprojplot(theta=q_vector_2[0], phi=q_vector_2[1], marker="*", color="lightskyblue", markersize=15);
    plt.tight_layout()

    return

#Define dipole + quadrupole prior
def dipole_quad_prior(cube):
    params = cube.copy()
    params[0] = cube[0]*200 # 0 - 200 (Nbar)
    params[1] = cube[1]/10 # 0 - 0.1 (D)
    params[2] = cube[2]/5 # 0 - 0.2 (Q)
    
    params[3] = cube[3]*2*np.pi # 0 - 2pi (l)
    c = cube[4] # 0 - 1, used to calculate theta below. 
    params[4] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b)
    
    params[5] = cube[5]*2*np.pi # 0 - 2pi (l1)
    c = cube[6] # 0 - 1, used to calculate theta below. 
    params[6] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b1)
    
    params[7] = cube[7]*2*np.pi # 0 - 2pi (l2)
    c = cube[8] # 0 - 1, used to calculate theta below. 
    params[8] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b2)
    return params

def vectorised_dipole_quad_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*200 # 0 - 200 (Nbar)
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

# Define dipole + quadrupole likelihood
def dipole_quad_likelihood(params):
    N_bar, D, Q, l, b, l1, b1, l2, b2 = params

    dipole_vec = hp.ang2vec(b, l)
    pixels = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX)))
    # pixel_vec = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX))) # check .T here for shapes. Vectorised is fixed
    
    dot_product = np.dot(dipole_vec, pixels)  
    angles = np.arccos(dot_product)
    dipole_signal = D*np.cos(angles)
    # angles = pixel_angles(pixel_vec, dipole_vec)
    
    v1 = hp.ang2vec(b1, l1)
    v2 = hp.ang2vec(b2, l2)
    
    Q_prime = np.outer(v1, v2)
    Q_star = 1/2 * (Q_prime + Q_prime.T)
    Q_hat = Q_star - (np.trace(Q_star)/3 * np.eye(3))

    quad_signal = Q*np.einsum('ij,i...,j...', Q_hat, pixels, pixels)
    
    lambda_i = N_bar * (1 + dipole_signal + quad_signal)
    
    return np.sum(poisson.logpmf(m, mu=lambda_i))

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
    

# Run the model on the data
dipole_param_names = [r'$\bar N$', 'D', r'$\ell$', r'$b$']
quad_param_names = [r'$\bar N$', 'Q', r'$\ell_1$', r'$b_1$', r'$\ell_2$', r'$b_2$']
dipole_quad_param_names = [r'$\bar N$', 'D', 'Q', r'$\ell$', r'$b$', r'$\ell_1$', r'$b_1$', r'$\ell_2$', r'$b_2$']

N_bars = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100] 

# Select the model to run and whether it is vectorised
dipole = True
quadrupole = False
dipole_and_quadrupole = False

vectorised = True

# Set seed and parameters
np.random.seed(42)
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
dipole_theta = np.radians(48)
dipole_phi = -np.radians(96)
dipole_vector = [dipole_theta, dipole_phi]
quad_vector_1 = [np.pi/4, np.pi*4/5]
quad_vector_2 = [np.pi*3/4 , -np.pi/4]

# Run the chosen model
if dipole == True:
    for N in N_bars:
        # Generate the signal
        np.random.seed(42)
        m, lambda_true_array = dipole_sampling(NSIDE=64,  N_bar=N, D=0.007, dipole_theta=dipole_theta,
                                                dipole_phi=dipole_phi)
        
        # Define the sampler, vectorised or not
        if vectorised == False:
            dipole_sampler = ultranest.ReactiveNestedSampler(dipole_param_names, dipole_likelihood,
                                                        dipole_prior, log_dir=f'../log_dir/testing/dipole/regular/nbar_{N}',
                                                        vectorized=False, resume=True)
        else:
            dipole_sampler = ultranest.ReactiveNestedSampler(dipole_param_names, vectorised_dipole_likelihood,
                                                        vectorised_dipole_prior, log_dir=f'../log_dir/testing/dipole/vectorised/nbar_{N}',
                                                        vectorized=True, resume=True)
        # Run the sampler
        dipole_result = dipole_sampler.run()
        
elif quadrupole == True:
    for N in N_bars:
        np.random.seed(42)
        # Generate the signal
        m, lambda_true_array = quadrupole_sampling(NSIDE=64,  N_bar=N, Q=0.014,
                                                vector_1=quad_vector_1, vector_2=quad_vector_2)
        
        # Define the sampler, vectorised or not
        if vectorised == False:
            quadrupole_sampler = ultranest.ReactiveNestedSampler(quad_param_names, quadrupole_likelihood,
                                                        quadrupole_prior, log_dir=f'../log_dir/testing/quadrupole/regular/nbar_{N}',
                                                        vectorized=False, resume=True)
        else:
            quadrupole_sampler = ultranest.ReactiveNestedSampler(quad_param_names, vectorised_quadrupole_likelihood,
                                                        vectorised_quadrupole_prior, log_dir=f'../log_dir/testing/quadrupole/vectorised/nbar_{N}',
                                                        vectorized=True, resume=True)

        # Set the stepsampler
        quadrupole_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=10,
            generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))
        # Run the sampler
        quadrupole_sampler = quadrupole_sampler.run()
    
elif dipole_and_quadrupole == True:
    for N in N_bars:
        np.random.seed(42)
        # Generate the signal
        m, lambda_true_array = dipole_quad_sampling(NSIDE=64,  N_bar=N, D=0.007, Q=0.014,
                                                    d_vector=dipole_vector, q_vector_1=quad_vector_1, q_vector_2=quad_vector_2)
        # Define the sampler, vectorised or not
        if vectorised == False:
            dipole_quad_sampler = ultranest.ReactiveNestedSampler(dipole_quad_param_names, dipole_quad_likelihood,
                                                    dipole_quad_prior, log_dir=f'../log_dir/testing/dipole_quadrupole/regular/nbar_{N}',
                                                    vectorized=False, resume=True)
        else:   
            dipole_quad_sampler = ultranest.ReactiveNestedSampler(dipole_quad_param_names, vectorised_dipole_quad_likelihood,
                                                    vectorised_dipole_quad_prior, log_dir=f'../log_dir/testing/dipole_quadrupole/vectorised/nbar_{N}',
                                                    vectorized=True, resume=True)
        # Set the stepsampler
        dipole_quad_sampler.stepsampler = ultranest.stepsampler.SliceSampler(
            nsteps=10,
            generate_direction = (ultranest.stepsampler.generate_mixture_random_direction))
        # Run the sampler
        dipole_quad_result = dipole_quad_sampler.run()
    
    

