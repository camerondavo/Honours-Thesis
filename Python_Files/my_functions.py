#Imports
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
# from mpi4py import MPI
import gc
import json
import corner
import os

#Functions

#### SAMPLING FUNCTIONS ####
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


#### PLOTTING FUNCTIONS ####       
def dipole_plotting(m, dipole_theta=None, dipole_phi=None, title=None, unit=None, cmap='plasma', vmin=None, vmax=None):
    projview(m, title=title, unit=unit,
    graticule=True, graticule_labels=True, projection_type="mollweide", cmap=cmap, min=vmin, max=vmax);

    newprojplot(theta=dipole_theta, phi=dipole_phi, marker="*", color="k", markersize=15);
    plt.tight_layout()
    return

def quadrupole_plotting(m, vector_1=None, vector_2=None, title=None, unit=None, cmap='plasma'):
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

def dipole_quad_plotting(m, d_vector=None, q_vector_1=None, q_vector_2=None, title=None, unit=None, cmap='plasma'):
    projview(m, title=title, unit=unit,
    graticule=True, graticule_labels=True, projection_type="mollweide", cmap=cmap);
    # adjusting the angles to be in the range [-pi, pi] for plotting
    if q_vector_1[1] > np.pi:
        q_vector_1[1] = q_vector_1[1]- 2*np.pi
    if q_vector_2[1] > np.pi:
        q_vector_2[1] = q_vector_2[1]- 2*np.pi  
    newprojplot(theta=d_vector[0], phi=d_vector[1], marker="*", color="b", markersize=15);
    newprojplot(theta=q_vector_1[0], phi=q_vector_1[1], marker="*", color="lightskyblue", markersize=15);
    newprojplot(theta=q_vector_2[0], phi=q_vector_2[1], marker="*", color="lightskyblue", markersize=15);
    plt.tight_layout()

    return

## Contour plotting ##
def omega_to_theta(omega):
    '''Convert solid angle in steradins to theta in radians for
    a cone section of a sphere. Taken from the code used in Secrest (2021),
    namely in CatWISE_Dipole_Resulys.ipynb. '''
    return np.arccos(1 - omega / (2 * np.pi)) * u.rad

def compute_smooth_map(m: np.ndarray, weights=None, angle_scale=1):
        'Smooth the map using a moving average.'
        included_pixels = np.where(~np.isnan(m))[0]
        smoothed_map = np.full(m.shape, np.nan, dtype=float)
        nside = hp.get_nside(m)
        
        if type(weights) != np.ndarray:
            weights = np.ones_like(m).astype('float')

        smoothing_radius = omega_to_theta(angle_scale).value
        for p_index in included_pixels:
            vec = hp.pix2vec(nside, p_index)
            disc = hp.query_disc(nside, vec, smoothing_radius)
            smoothed_map[p_index] = np.nanmean(m[disc].astype(float) * weights[disc].astype(float))

        return smoothed_map
    
def sigma_to_prob2D(sigmas):
    return 1 - np.exp(-0.5 * np.array(sigmas) ** 2)

def compute_2D_contours(P_xy, contour_levels):
    '''
    Compute contour heights corresponding to sigma levels of probability
    density by creating a mapping (interpolation function) from the CDF
    (enclosed prob) to some arbitrary level of probability density.
    from here: https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution

    :param P_xy: normalised 2D probability (not density)
    :param contour_levels: pass list of sigmas at which to draw the contours
    :return:
        1. vector of probabilities corresponding to heights at which to
        draw the contours (pass to e.g. plt.contour with levels= kwarg);
        2. uniformly spaced probability levels between 0 and max prob
        3. CDF at given P_level, of length 1000 (the hardcoded number of
            P_levels)
    '''
    P_levels = np.linspace(0, P_xy.max(), 2000)
    mask = (P_xy >= P_levels[:, None, None])
    P_integral = (mask * P_xy).sum(axis=(1,2))
    f = interp1d(P_integral, P_levels)
    t_contours = np.flip(f(sigma_to_prob2D(contour_levels)))
    return t_contours, P_levels, P_integral

def dipole_direction_distribution(result, dipole_theta, dipole_phi):
    equal_weighted_samples = resample_equal(
        np.array(result['samples']), np.array(result['weighted_samples']['weights']))
    nside=64
    npix = hp.nside2npix(nside)
    healpix_map = np.zeros(npix)
    l = equal_weighted_samples[:, 2]  
    b = equal_weighted_samples[:, 3]

    # Bin (l,b) into healpy pixels
    pixels = hp.ang2pix(nside, b,l, nest=False)
    for pix in pixels:
        healpix_map[pix] += 1
    
    #Smooth distribution
    smooth = 0.1
    healpix_map /= np.sum(healpix_map)
    smooth_map = hp.sphtfunc.smoothing(healpix_map, sigma=smooth)
    smooth_map[smooth_map < 0] = 0
    smooth_map /= np.sum(smooth_map)
    
    # Compute contours
    X, Y, proj_map = hp.projview(
    smooth_map, return_only_data=True, xsize=2000)
    P_xy = proj_map / np.sum(proj_map)
    contour_levels = [0.5, 1, 1.5]  
    t_contours, P_levels, P_integral = compute_2D_contours(P_xy, contour_levels)
    
    #plot
    hp.projview(smooth_map, cmap="Blues", graticule=True, graticule_labels=True)
    newprojplot(theta=dipole_theta, phi=dipole_phi, marker="*", color="r", markersize=15, zorder=5);

    plt.contour(X, Y, P_xy, levels=t_contours, colors=['C0','b','navy'])

    plt.tight_layout()
    
    return

def process_vector_component(l, b, nside, smooth_sigma):
        pixels = hp.ang2pix(nside, b, l, nest=False)
        healpix_map = np.bincount(pixels, minlength=hp.nside2npix(nside)).astype(np.float64)
        healpix_map /= healpix_map.sum()
        smoothed_map = hp.sphtfunc.smoothing(healpix_map, sigma=smooth_sigma)
        smoothed_map[smoothed_map < 0] = 0
        smoothed_map /= smoothed_map.sum()
        return smoothed_map
        
def get_projection_and_contours(hmap, contour_levels=[0.5, 1, 1.5, 2]):
        X, Y, proj_map = hp.projview(hmap, return_only_data=True, xsize=2000)
        P_xy = proj_map / proj_map.sum()
        t_contours, P_levels, P_integral = compute_2D_contours(P_xy, contour_levels)
        return X, Y, P_xy, t_contours

def mask_data(NSIDE, m, mask_angle):
    NPIX = hp.nside2npix(NSIDE)
    theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
    mask = (theta >= np.radians(90 - mask_angle)) & (theta <= np.radians(90+mask_angle))
    
    # for plotting, need to use nans
    m_plot = m.copy()
    m_plot = m_plot.astype(float)  # convert to float so np.nan is allowed
    m_plot[mask] = np.nan

    # for fitting, can't use nans need to remove the masked indices. Also need fit_mask to mask in likelihood too
    m_fit = m.copy()
    fit_mask = ~mask
    m_fit = m[fit_mask]

    return m_plot, m_fit, fit_mask

def partial_sky_masking(m, theta, phi, r=40): 
    smooth_map = compute_smooth_map(m, angle_scale=1)
    vec = hp.ang2vec(theta, phi)
    ipix_disc = hp.query_disc(nside=NSIDE, vec=vec, radius=np.radians(r))

    mask = np.zeros(NPIX, dtype=bool)
    mask[ipix_disc] = True  # Set the disc pixels to True

    # For plotting, we need the array to be of length NPIX
    m_plot = hp.ma(m.copy().astype(float))
    m_plot.mask = ~mask

    m_plot_smooth = hp.ma(smooth_map.copy().astype(float))
    m_plot_smooth.mask = ~mask

    # for fitting, can't use nans need to remove the masked indices. Also need fit_mask to mask in likelihood too
    m_fit = m.copy()
    fit_mask = mask
    m_fit = m[fit_mask]

    return m_plot, m_plot_smooth, m_fit, fit_mask

#### Prior Functions ####

def monopole_prior(cube):
    params = cube.copy()
    params[0] = cube[0]*200 # 0 - 200 (Nbar)
    
    return params

def vectorised_monopole_prior(cube):
    params = cube.copy()
    params[:,0] = cube[:,0]*220 
    
    return params

def dipole_prior(cube):
    params = cube.copy()
    params[0] = cube[0]*200 # 0 - 200
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



#### Likelihood Functions ####
def monopole_likelihood(params):
    N_bar = params[0]
    
    lambda_i = N_bar * np.ones(NPIX)
    
    return np.sum(poisson.logpmf(m, mu=lambda_i))

def vectorised_monopole_likelihood(params):
    N_bar = params[:,0]    
    lambda_i = (N_bar[None, :] * np.ones((NPIX, 1)))
    return np.sum(poisson.logpmf(m, mu=lambda_i.T), axis=1)

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

