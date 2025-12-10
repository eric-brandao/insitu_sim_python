# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 14:19:12 2025

@author: Win11
"""
import pickle
import numpy as np
import scipy
from tqdm import tqdm
# import matplotlib.tri as tri
from receivers import Receiver
import matplotlib.pyplot as plt
#from sklearn.linear_model import Ridge
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# from scipy.special import roots_legendre, roots_laguerre
# from scipy.sparse.linalg import lsmr
#from lcurve_functions_EU import csvd, l_curve, tikhonov
import lcurve_functions as lc
import utils_insitu as ut_is


class RegTamura(object):
    """ Regularized Tamura method.

    Attributes
    ----------
    controls : object (AlgControls)
        Controls of the decomposition (frequency spam)
    receivers : object (Receiver)
        The receivers in the field - this contains the information of the coordinates of
        the microphones in your array
    
    Methods
    ----------
    
    """

    def __init__(self, p_mtx=None, controls=None, receivers=None, source = None, 
                 k_stop_fac = 2, delta_k = 0.1, delta_r = 0.04, factor = 1.5,
                 regu_par = 'L-curve'):
        """
        Parameters
        ----------
        p_mtx : (N_rec x N_freq) numpy array
            A matrix containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)

        receivers : object (Receiver)
            The receivers in the field
        source : object (Receiver)
            The source in the field
        k_stop_fac : float
            How far you want to go with you wavenumber spectrum. This will be a multiple of k0.
            Should be at least 1.0 (default is 2)
        delta_k : wave-number resolution
        delta_r : used to estimate scaling planes position
        regu_par : str
            Automatic choice of regularization parameter. Default is "L-curve". It can
            be "L-curve" or l-curve for L-curve choice; or "gcv" or "GCV" for generalized
            cross-validation. Any other choice reverts do default.

        The objects are stored as attributes in the class (easier to retrieve).
        """

        self.pres_s = p_mtx
        self.controls = controls
        self.receivers = receivers
        self.source = source
        if regu_par == 'L-curve' or regu_par == 'l-curve':
            self.regu_par_fun = lc.l_curve
            # print("You choose L-curve to find optimal regularization parameter")
        elif regu_par == 'gcv' or regu_par == 'GCV':
            self.regu_par_fun = lc.gcv_lambda
            # print("You choose GCV to find optimal regularization parameter")
        elif regu_par == 'ncp' or regu_par == 'NCP':
            self.regu_par_fun = lc.ncp
            # print("You choose NCP to find optimal regularization parameter")
        else:
            self.regu_par_fun = lc.l_curve
            # print("Returning to default L-curve to find optimal regularization parameter")
        
        self.k_stop_fac = k_stop_fac
        self.delta_k = delta_k
        self.delta_r = delta_r
        self.factor = factor
        self.get_scaling_planes()
        
    def get_scaling_planes(self):
        """ Computes scaling planes positioning
        """
        self.z_minus = - self.factor * self.delta_r
        z_top = np.amin(self.receivers.coord[:,2]) + np.amax(self.receivers.coord[:,2])
        self.z_plus = z_top + self.factor * self.delta_r
            
    def get_k_vec(self, k0):
        """ Form a k vector
        """
        kr_max = np.pi/0.01
        # k = np.arange(0, kr_max + self.delta_k, self.delta_k)
        
        k = np.arange(self.delta_k/2, kr_max + self.delta_k/2, self.delta_k)
        return k
    
    def get_k0z(self, k0):
        """ Form a k0z vector
        """
        k = self.get_k_vec(k0)
        k0z = np.sqrt(k0**2 - k**2 + 1j*0)
        return k0z
    
    def get_rm(self,):
        """ Get radial coordinates of the array
        """
        rm = np.sqrt(self.receivers.coord[:,0]**2 + self.receivers.coord[:,1]**2)
        return rm
    
    def psi_inc_z(self, k0z, zm):
        """ Incident part kernel (z)
        """
        psi_inc = np.exp(-1j*np.outer((zm - self.z_plus), k0z))
        return psi_inc
    
    def psi_ref_z(self, k0z, zm):
        """ Reflected part kernel (z)
        """
        psi_ref = np.exp(1j*np.outer((zm - self.z_minus), k0z))
        return psi_ref
    
    def psi_c(self, k, rm):
        """ Circular part kernel
        """
        bessel_0 = scipy.special.jv(0, np.outer(rm, k))
        #psi_circ = np.outer(rm, np.ones(bessel_0.shape[1])) * bessel_0 #k * self.delta_k
        psi_circ = bessel_0 * k * self.delta_k
        return psi_circ
    
    def psi_mtx(self, k0):
        """ Form system matrix (for sound field separation)
        """
        zm = self.receivers.coord[:,2]
        rm = self.get_rm()
        k = self.get_k_vec(k0)
        k0z = self.get_k0z(k0)
        psi_circ = self.psi_c(k, rm)
        psi_i_z = self.psi_inc_z(k0z, zm)
        psi_r_z = self.psi_ref_z(k0z, zm)
        h_mtx = np.hstack((psi_i_z * psi_circ,
                           psi_r_z * psi_circ))
        return h_mtx, k
    
    def pkr(self, id_f = 0):
        """ regularized total WNS
        """
        h_mtx, k = self.psi_mtx(self.controls.k0[id_f])
        u, sig, v = lc.csvd(h_mtx)
        lambd_value = lc.l_curve(u, sig, self.pres_s[:, id_f])
        pkr = lc.tikhonov(u, sig, v, self.pres_s[:, id_f], lambd_value)
        return pkr, k
    
    def psi_mtx_tot(self, k0):
        """ Form system matrix (for total wns)
        """
        # zm = self.receivers.coord[:,2]
        rm = self.get_rm()
        k = self.get_k_vec(k0)
        # k0z = self.get_k0z(k0)
        psi_circ = self.psi_c(k, rm)
        # psi_i_z = self.psi_inc_z(k0z, zm)
        # psi_r_z = self.psi_ref_z(k0z, zm)
        h_mtx = psi_circ
        return h_mtx, k
    
    def pkr_total(self, id_f = 0):
        """ regularized total WNS
        """
        h_mtx, k = self.psi_mtx_tot(self.controls.k0[id_f])
        u, sig, v = lc.csvd(h_mtx)
        lambd_value = lc.l_curve(u, sig, self.pres_s[:, id_f])
        pkr = lc.tikhonov(u, sig, v, self.pres_s[:, id_f], lambd_value)
        return pkr, k
    
    def my_fb_matrix(self, pres_rec, dr = 0.01):
        """ Computes a Fourier-Bessel matrix for direct discrete transform
        """
        # Num of samples
        nfft = len(pres_rec)
        # Sampling rate
        fr = np.pi/dr
    
        m = np.arange(nfft)
        n = np.arange(nfft)
        
        krm = np.linspace(0, fr*(nfft-1)/nfft, nfft)
        
        Fmn = np.zeros((nfft,nfft), dtype = complex)
        for mm in m:
            for nn in n:
                Fmn[mm,nn] = 2*np.pi*dr*nn*(scipy.special.jv(0,krm[mm]*nn*dr))    
        return Fmn, krm
    
    def fourier_bessel(self, id_f = 0, dr = 0.01):
        """ Computes Fourier Bessel transform with my_fb_matrix function
        """
        # data
        pres_rec = self.pres_s[:, id_f]
        # Form matrix
        Fmn, krm = self.my_fb_matrix(pres_rec, dr)
        # Compute spectrum
        Pkr = (Fmn @ pres_rec)#/nfft
        return Pkr, krm

def compute_ref_coef(Pk1, Pk2, z1, z2, kr, k0):
    """ Tamura's way to compute the reflection coefficient. From Fourier-Bessel spectrum
    """
    # propagating part of kr
    kr_prop = kr[kr<=k0]
    Pk1 = Pk1[kr<=k0]
    Pk2 = Pk2[kr<=k0]
    # angles of incidence in kr
    theta = np.arcsin(kr_prop/k0)
    # kz
    kz = np.sqrt(k0**2 - kr_prop**2)
    # init
    Ref_coef = np.zeros(len(kr_prop), dtype = complex)
    # compute
    Pi = (Pk1 * np.exp(1j*kz*z2) - Pk2 * np.exp(1j*kz*z1))#/(2*1j*np.sin(kz*(z2-z1)))
    Pr = (Pk2 * np.exp(-1j*kz*z1) - Pk1 * np.exp(-1j*kz*z2))#/(2*1j*np.sin(kz*(z2-z1)))
    Ref_coef = Pr/Pi
    return Ref_coef, theta        
        
        
    
    
        