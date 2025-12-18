# -*- coding: utf-8 -*-
"""
Created on October 21 2025

@author: Eric Brandão
"""

import numpy as np
# import matplotlib.pyplot as plt
# import scipy 
from tqdm import tqdm
import utils_insitu as ut_is
# from receivers import Receiver
# from sources import Source
# from material import PorousAbsorber

class dDCISM(object):
    """ Direct Discrete Image Source Method
    
    As implemented by M. Eser, JASA 2021
    """
    def __init__(self, air = None, controls = None, material = None, 
                 source = None, receivers = None, 
                 T0 = 7.5, dt = 0.1, tol = 1e-6, gamma=1.0):
        
        """ Initialization of object:
            
        Parameters
        -------------
        air : AirProperties object
        controls : Controls object
        material : PorousMaterial object
        sources : Source object
        receivers : Receiver object
        T0 : float
            End value of integration path
        dt : float
            Sample rate at which reflection coeffcient is sampled
        tol : float
            Truncation threshold for SVD in MP
        gamma : float
            Scaling factor for extended range analysis
        """
        
        self.air = air
        self.controls = controls
        self.material = material
        self.receivers = receivers
        self.source = source
        self.T0 = T0
        self.dt = dt
        self.tol = tol
        self.gamma = gamma
        
        self.get_pencil_parameter()
        self.get_modified_sampling_path()
        self.r, self.hs, self.zr, self.r1, self.r2 = self.get_rec_parameters()
        
    def get_rec_parameters(self, ):
        """ Get receiver parameters
        
        Compute important receiver parameters such as source height, horizontal distance, etc
        
        Returns
        -------------
        r : numpy1dArray
            Horizontal distances
        zr : numpy1dArray
            Source height
        zr : numpy1dArray
            Receiver heights
        r1 : numpy1dArray
            Source to receiver distances
        r2 : numpy1dArray
            Imagge-Source to receiver distances
        """
        
        r = ((self.source.coord[0, 0] - self.receivers.coord[:,0]) ** 2 +\
             (self.source.coord[0, 1] - self.receivers.coord[:,1]) ** 2) ** 0.5  # Horizontal distance 
        hs = self.source.coord[0,2]
        zr = self.receivers.coord[:,2]  # Receiver height
        r1 = (r ** 2 + (self.source.coord[0, 2] - zr) ** 2) ** 0.5  # Euclidean dist. related to the real source
        r2 = (r ** 2 + (self.source.coord[0, 2] + zr) ** 2) ** 0.5  # Euclidean dist. related to the image source
        return r, hs, zr, r1, r2
           
    def get_pencil_parameter(self):
        """Caldulate pencil parameter according to Sarkar.1995 
        
        Returns
        -------------
        pencil_parameter : int
            Pencil parameter
        """
        
        self.pencil_parameter = int((self.T0 / (self.gamma * self.dt)) / 2 - 1)
        # return pencil_parameter
    
    def get_modified_sampling_path(self,):
        """ real-valued variable for modified sampling path """
        self.sampling_path = np.linspace(0, self.T0 / self.gamma, 
                                         int(self.T0 / (self.gamma * self.dt)))
    
    def get_vertical_wavenum(self, k):
        """ vertical wave number above sample; shape(len(f),len(t))
        modified sampling path Eq. (11) in Eser, 2021
        
        Parameters
        -------------
        k : float
            wave-number magnitude in [rad/m]
            
        Returns
        -------------
        kz : numpy1dArray
            Vertical wave-number
        """
        kz = self.gamma * k * (1j*self.sampling_path-1.0)
        return kz
    
    def sample_Vp_lr(self, k, beta):
        """ Sample the locally reacting reflection coefficient.
        
        This method samples the reflection coefficient. It applies to
        a locally reacting sample. 
        The material will have a complex surface admittance.
        
        The sampling is used to estimate the complex sources locations
        (in the complex plane) and amplitudes.
        
        Parameters
        -------------
        k : float
            wave-number magnitude in [rad/m]
        beta : complex
            surface adimmitance        
        Returns
        -------------
        Vp_sampled : numpy1dArray
            Sampled version of the reflection coefficient
        """
        # get Kz0 (vertical wave-number above layer)
        kz0 = self.get_vertical_wavenum(k = k)
        num = kz0 - k*beta
        den = kz0 + k*beta
        Vp_sampled = num / den
        return Vp_sampled
    
    def sample_Vp_nlr_layer(self, k, k_p, rho_p, t_p):
        """ Sample the Non-locally reacting reflection coefficient.
        
        This method samples the reflection coefficient. It applies to
        a layer of thickness "t_p" of a non-locally reacting sample above rigid
        backing. The material will have a characteristic wave-number, k_p, and
        a characteristic density, rho_p (both complex).
        
        The sampling is used to estimate the complex sources locations
        (in the complex plane) and amplitudes.
        
        Parameters
        -------------
        k : float
            wave-number magnitude in [rad/m]
        k_p : complex
            characteristic wave-number of the layer
        rho_p : complex
            characteristic density of the layer
        t_p : float
            thickness of the layer
        
        Returns
        -------------
        Vp_sampled : numpy1dArray
            Sampled version of the reflection coefficient
        """
        # get Kz0 (vertical wave-number above layer)
        kz0 = self.get_vertical_wavenum(k = k)
        # compute kza (vertical wave-number inside the layer)  
        kz1 = np.sqrt(k_p**2 - k**2 + kz0**2)
        # kz1 = np.sqrt(k_p**2 - k**2)
        num = (1j * kz0 / self.air.rho0) - (kz1 / rho_p) * np.tan(kz1 * t_p)
        den = (1j * kz0 / self.air.rho0) + (kz1 / rho_p) * np.tan(kz1 * t_p)
        Vp_sampled = num / den
        return Vp_sampled
    
    def compute_AnBn(self, Vp_sampled):
        """ Compute amplitudes and locations (transformed space) using MPM
        
        Compute An (amplitudes) and Bn (complex locations) in transformed space
        using the Matrix Pencil Method as implemented by Martin Eser (JASA 2021).
        
        Parameters
        -------------
        Vp_sampled : numpy1dArray
            Sampled version of the reflection coefficient
            
        Returns
        -------------
        An : numpy1dArray
            Compute An (amplitudes) in transformed space
        Bn : numpy1dArray
            Compute Bn (locations) in transformed space
        """
        An, Bn = ut_is.MPM(samp = Vp_sampled, L = self.pencil_parameter, 
                           Ts = self.sampling_path[1] - self.sampling_path[0], 
                           tol = self.tol)
        return An, Bn
    
    def compute_anbn(self, An, Bn, k):
        """ Compute an (amplitudes) and bn (complex locations) in the correct space.
        
        Parameters
        -------------
        An : numpy1dArray
            Compute An (amplitudes) in transformed space
        Bn : numpy1dArray
            Compute Bn (locations) in transformed space
        k : float
            wave-number magnitude in [rad/m]
            
        Returns
        -------------
        an : numpy1dArray
            Compute an (amplitudes) in correct space
        bn : numpy1dArray
            Compute bn (locations) in correct space
        """
        an = An * np.exp(-1j * Bn)
        bn = (1j * Bn) / (self.gamma * k)
        return an, bn
    
    def kernel_p(self, k, r):  # pressure kernel function
        iq = (np.exp(-1j * k * r)) / r
        return iq
    
    def get_cplx_rec_parameters(self, an, bn):
        """ Complex receiver parameters in matrix form
        
        Parameters
        -------------
        an : numpy1dArray
            Compute an (amplitudes) in correct space
        bn : numpy1dArray
            Compute bn (locations) in correct space        
        """
        zr_mtx = np.repeat(np.array([self.zr]), len(an), axis = 0).T
        r_mtx = np.repeat(np.array([self.r]), len(an), axis = 0).T
        bn_mtx = np.repeat(np.array([bn]), len(self.r), axis = 0)
        # print(bn_mtx.shape)
        dn = np.sqrt(r_mtx**2 + (self.hs + zr_mtx + 1j*bn_mtx)**2)
        return dn
    
    def predict_p(self, k, Vp_sampled):
        """ Uses MPM to predict the sound pressure
        
        Valid for any material configuration
        
        Parameters
        -------------
        k : float
            wave-number magnitude in [rad/m]
        Vp_sampled : numpy1dArray
            Sampled version of the reflection coefficient
        """
        # Compute source amplitudes
        An, Bn = self.compute_AnBn(Vp_sampled = Vp_sampled)
        an, bn = self.compute_anbn(An, Bn, k = k)
        dn = self.get_cplx_rec_parameters(an, bn)
        # Compute sound pressure for all receivers  
        g_mtx = self.kernel_p(k, dn)
        pcim = g_mtx @ an
        pres = self.kernel_p(k, self.r1) + pcim
        return pres
    
    def predict_p_lr(self, k, beta):
        """ Predict sound pressure at all receivers (single frequency)
        
        Parameters
        -------------
        k : float
            wave-number magnitude in [rad/m]
        beta : complex
            surface adimmitance   
        
        Returns
        -------------
        pres : numpy1dArray
            complex sound pressure at all receivers.
        """
        # Sample Vp
        Vp_sampled = self.sample_Vp_lr(k = k, beta = beta)
        # Compute source amplitudes
        pres = self.predict_p(k, Vp_sampled)
        return pres
    
    def predict_p_spk_lr(self):
        """ Predict the sound pressure spectra
        
        If a material object is passed, this is used to predict the sound field
        spectrum at all receivers. This is useful for numerical simulations of
        the sound field (but not so much inverse problems)
        """
        # Initalize pressure matrix
        self.pres_mtx = np.zeros((self.receivers.coord.shape[0], 
                             len(self.controls.k0)), dtype = complex)
        # Initialize bar
        bar = tqdm(total=len(self.controls.k0),
                   desc='Calculating pressure spectra', ascii=False)
        
        # Freq loop
        for jk, k0 in enumerate(self.controls.k0):
            beta = (self.air.rho0 * self.air.c0)/self.material.Zs[jk]
            # beta = 1/self.material.Zs[jk]
            # beta = 0
            self.pres_mtx[:, jk] = self.predict_p_lr(k = k0,
                                                     beta = beta)
            bar.update(1)
        bar.close()
    
    def predict_p_nlr_layer(self, k, k_p, rho_p, t_p):
        """ Predict sound pressure at all receivers (single frequency)
        
        Parameters
        -------------
        k : float
            wave-number magnitude in [rad/m]
        k_p : complex
            characteristic wave-number of the layer
        rho_p : complex
            characteristic density of the layer
        t_p : float
            thickness of the layer
        
        Returns
        -------------
        pres : numpy1dArray
            complex sound pressure at all receivers.
        """
        # Sample Vp
        Vp_sampled = self.sample_Vp_nlr_layer(k = k, k_p = k_p, 
                                  rho_p = rho_p, t_p = t_p)
        # Compute source amplitudes
        pres = self.predict_p(k, Vp_sampled)
        return pres
    
    def predict_p_spk_nlr_layer(self):
        """ Predict the sound pressure spectra
        
        If a material object is passed, this is used to predict the sound field
        spectrum at all receivers. This is useful for numerical simulations of
        the sound field (but not so much inverse problems)
        """
        # Initalize pressure matrix
        self.pres_mtx = np.zeros((self.receivers.coord.shape[0], 
                             len(self.controls.k0)), dtype = complex)
        # Initialize bar
        bar = tqdm(total=len(self.controls.k0),
                   desc='Calculating pressure spectra', ascii=False)
        
        # Freq loop
        for jk, k0 in enumerate(self.controls.k0):
            self.pres_mtx[:, jk] = self.predict_p_nlr_layer(k = k0, 
                                                            k_p = self.material.kp[jk],
                                                            rho_p = self.material.rhop[jk],
                                                            t_p = self.material.thickness)
            bar.update(1)
        bar.close()
        #return self.pres_mtx
    
    def add_noise(self, snr = 30, uncorr = False):
        """ Add gaussian noise to the simulated data.

        The function is used to add noise to the pressure and particle velocity data.
        it reads the clean signal and estimate its power. Then, it estimates the power
        of the noise that would lead to the target SNR. Then it draws random numbers
        from a Normal distribution with standard deviation =  noise power

        Parameters
        ----------
        snr : float
            The signal to noise ratio you want to emulate
        uncorr : bool
            If added noise to each receiver is uncorrelated or not.
            If uncorr is True the the noise power is different for each receiver
            and frequency. If uncorr is False the noise power is calculated from
            the average signal magnitude of all receivers (for each frequency).
            The default value is False
        """
        signal = self.pres_mtx
        try:
            signal_u = self.uz_s[0]
        except:
            signal_u = np.zeros(1)
        if uncorr:
            signalPower_lin = (np.abs(signal)/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
        else:
            # signalPower_lin = (np.abs(np.mean(signal, axis=0))/np.sqrt(2))**2
            signalPower_lin = ((np.mean(np.abs(signal), axis=0))/np.sqrt(2))**2
            signalPower_dB = 10 * np.log10(signalPower_lin)
            noisePower_dB = signalPower_dB - snr
            noisePower_lin = 10 ** (noisePower_dB/10)
            if signal_u.any() != 0:
                signalPower_lin_u = (np.abs(np.mean(signal_u, axis=0))/np.sqrt(2))**2
                signalPower_dB_u = 10 * np.log10(signalPower_lin_u)
                noisePower_dB_u = signalPower_dB_u - snr
                noisePower_lin_u = 10 ** (noisePower_dB_u/10)
        np.random.seed(0)
        noise = np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin), size = signal.shape)
        # noise = 2*np.sqrt(noisePower_lin)*\
        #     (np.random.randn(signal.shape[0], signal.shape[1]) + 1j*np.random.randn(signal.shape[0], signal.shape[1]))
        self.pres_mtx = signal + noise
        if signal_u.any() != 0:
            # print('Adding noise to particle velocity')
            noise_u = np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape) +\
                1j*np.random.normal(0, np.sqrt(noisePower_lin_u), size = signal_u.shape)
            self.uz_s[0] = signal_u + noise_u
    
    def save(self, filename = 'dDCISM', path = ''):
        """ To save the decomposition object as pickle
        """
        ut_is.save(self, filename = filename, path = path)

    def load(self, filename = 'dDCISM', path = ''):
        """ To load the decomposition object as pickle

        You can instantiate an empty object of the class and load a saved one.
        It will overwrite the empty object.
        """
        ut_is.load(self, filename = filename, path = path)
        
    
        
        
        

