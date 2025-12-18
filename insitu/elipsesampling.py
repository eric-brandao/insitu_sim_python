# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 08:29:34 2025

@author: Eric Brandao
"""
import numpy as np
# import scipy
# import matplotlib.pyplot as plt
# import utils_insitu as ut_is

class ElipsoidalSampling():
    """ Find elipsoidal region for sampling
    """
    
    def __init__(self, coords = None):
        self.coords = coords
        self.n_pts = self.coords.shape[0]
        self.Ndim = self.coords.shape[1]
    
    def mean(self,):
        """ Computes coordinate's mean value for each dimension
        """
        mu = np.mean(self.coords, axis = 0)
        return mu
    
    def subtract_mean(self, mu):
        """ mean subtraction from coordinates
        """
        coords_minus_mean = self.coords-mu
        return coords_minus_mean
    
    def covariance_mtx(self, coords_minus_mean):
        """ Computes the covariance matrix
        """
        cov_mtx = (coords_minus_mean.T @ coords_minus_mean) * (1/(self.n_pts-1))
        return cov_mtx
    
    def eigen_decomp(self, cov_mtx):
        """ Eigenvalue decomposition of covariance matrix
        """
        eigvals, eigvecs = np.linalg.eigh(cov_mtx)
        return eigvals, eigvecs
        
    def mahalanobis_dist(self, eigvals, eigvecs, coords_minus_mean, eps = 1e-12):
        """ Computes the mahalanobis distance to build elipse 
        
        Uses eigen decomposition
        """
        eigvals_clipped = np.clip(eigvals, eps, None)  # ensure >= eps
        # project into eigenbasis: y = Xc @ V  (each row is y_i)
        Y = coords_minus_mean.dot(eigvecs)
        # compute squared Mahalanobis distances: sum_j (y_ij^2 / lambda_j)
        d2 = np.sum((Y ** 2) / eigvals_clipped[None, :], axis=1)
        E = np.amax(d2)
        return E
        
    def ellipse_axis(self, E, eigvals, enlargement_factor = 1.1):
        """ Compute ellipse axis with enlargement
        """
        axes = np.sqrt(enlargement_factor * E * eigvals)
        return axes
        
    def sample_in_sphere(self,):
        """ Sample single point inside a unit sphere
        """
        u = np.random.normal(size = self.Ndim)
        u = u / np.linalg.norm(u)     # random direction on the unit sphere
        r = np.random.rand() ** (1.0/self.Ndim)  # correct radius distribution
        theta_unit = r * u                 # uniform inside unit ball
        return theta_unit
    
    def sample_in_ellipse(self, elargement_factor = 1.1, eps = 1e-12):
        """ Sample single point inside a the ellipse
        """
        theta_unit = self.sample_in_sphere()
        mu = self.mean()
        coords_minus_mean = self.subtract_mean(mu)
        cov_mtx = self.covariance_mtx(coords_minus_mean)
        eigvals, eigvecs = self.eigen_decomp(cov_mtx)
        E = self.mahalanobis_dist(eigvals, eigvecs, coords_minus_mean, eps = eps)
        axes = self.ellipse_axis(E, eigvals, elargement_factor = elargement_factor)
        theta_ell = mu + eigvecs @ (axes * theta_unit)
        return theta_ell
    
    def sample_multi_in_sphere(self, n_samples = 10):
        """ Sample multiple points inside a unit sphere
        """
        u = np.random.normal(size = (n_samples, self.Ndim))
        u /= np.linalg.norm(u, axis=1)[:, None]     # random direction on the unit sphere
        r = np.random.rand(n_samples) ** (1.0/self.Ndim)  # correct radius distribution
        theta_unit = u * r[:, None]                 # uniform inside unit ball
        return theta_unit
    
    def sample_multi_in_ellipse(self, n_samples = 10, enlargement_factor = 1.1, eps = 1e-12):
        """ Sample multiple points inside a the ellipse
        """
        theta_unit = self.sample_multi_in_sphere(n_samples = n_samples)
        mu = self.mean()
        coords_minus_mean = self.subtract_mean(mu)
        cov_mtx = self.covariance_mtx(coords_minus_mean)
        eigvals, eigvecs = self.eigen_decomp(cov_mtx)
        E = self.mahalanobis_dist(eigvals, eigvecs, coords_minus_mean, eps = eps)
        axes = self.ellipse_axis(E, eigvals, enlargement_factor = enlargement_factor)
        # theta_scaled = theta_unit * axes[None, :]
        theta_ell = (eigvecs @ (theta_unit * axes[None, :]).T).T + mu[None, :]
        return theta_ell
        
        