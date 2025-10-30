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

# Dipole
def dipole_prior_transform(cube):
    params = cube.copy()
    params[0] = cube[0]*200 # 0 - 100
    params[1] = cube[1]/10 # 0 - 0.1
    params[2] = cube[2]*2*np.pi # 0 - 2pi
    
    c = cube[3] # 0 - 1, used to calculate theta below. 
    # Oliver uses 0 - 0.1, but this doesnt return theta between (0,pi), is it a typo or am i missing something else?
    params[3] = np.arccos(np.clip(1 - 2*c, -1, 1)) #Clipping to ensure we stay in the valid range
    
    return params
    
def dipole_likelihood(params):
    N_bar, D, l, b = params
    pixel_vec = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX))).T
    dipole_vec = hp.ang2vec(b, l) # dipole vector in cartesian
    angles = pixel_angles(pixel_vec, dipole_vec)
    
    lambda_i = N_bar * (1 + D*np.cos(angles))
    
    return np.sum(poisson.logpmf(m, mu=lambda_i))


# Quadrupole
def quadrupole_prior_transform(cube):
    
    params = cube.copy()
    params[0] = cube[0]*200 # 0 - 2000 (Nbar)
    params[1] = cube[1]/5 # 0 - 0.2 (Q)
    params[2] = cube[2]*2*np.pi # 0 - 2pi (l1)
    
    c = cube[3] # 0 - 1, used to calculate theta below. 
    params[3] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b1)
    
    params[4] = cube[4]*2*np.pi # 0 - 2pi (l2)
    params[5] = np.arccos(np.clip(1 - 2*c, -1, 1)) #(b2)
    
    return params
    
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

# Dipole + Quadrupole
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

def dipole_quad_likelihood(params):
    N_bar, D, Q, l, b, l1, b1, l2, b2 = params

    dipole_vec = hp.ang2vec(b, l)
    pixels = hp.pix2vec(NSIDE, np.arange(NPIX))
    pixel_vec = np.vstack(hp.pix2vec(NSIDE, np.arange(NPIX))).T
    angles = pixel_angles(pixel_vec, dipole_vec)
    
    a = hp.ang2vec(b1, l1)
    b = hp.ang2vec(b2, l2)
    
    Q_prime = np.outer(a, b)
    Q_star = 1/2 * (Q_prime + Q_prime.T)
    Q_hat = Q_star - np.trace(Q_star)/3

    f = Q*np.einsum('ij,i...,j...', Q_hat, pixels, pixels)
    
    lambda_i = N_bar * (1 + D*np.cos(angles) + f)
    
    return np.sum(poisson.logpmf(m, mu=lambda_i))
