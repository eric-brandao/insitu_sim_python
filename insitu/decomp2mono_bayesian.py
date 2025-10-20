# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 2025

@author: Eric Brandão
"""

import numpy as np
from tqdm import tqdm
from receivers import Receiver
import matplotlib.pyplot as plt
#from sklearn.linear_model import Ridge
#from lcurve_functions_EU import csvd, l_curve, tikhonov
import lcurve_functions as lc
import utils_insitu as ut_is
from decomp2mono import Decomposition_2M
from baysian_sampling import BayesianSampler

class Decomposition_2M_Bayesian(Decomposition_2M):
    """ Bayesian Decomposition of the sound field using source/image-source monopoles.

    The class has methods to perform sound field decomposition into a set of 
    source and image source componensts.
    """
    def __init__(self, p_mtx=None, controls=None, receivers=None, source_coord=[0,0,0]):
        """

        Parameters
        ----------
        p_mtx : (N_rec x N_freq) numpy array
            A matrix containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)
        material : object (PorousAbsorber)
            Contains the material properties (surface impedance).
        receivers : object (Receiver)
            The receivers in the field

        The objects are stored as attributes in the class (easier to retrieve).
        """
        Decomposition_2M.__init__(self, p_mtx, controls, receivers, source_coord)
        super().__init__(p_mtx, controls, receivers, source_coord)
        self.parameters_names = ["Re(s)", "Im(s)", "Re(is)", "Im(is)"]
        # Get receiver data (frequency independent)
        self.r, self.zr, self.r1, self.r2 = self.get_rec_parameters(self.receivers)
    
    def set_prior_limits(self, lower_bounds = [0, 0, 0, 0], 
                         upper_bounds = [1, 1, 1, 1]):
        """ Set the uniform prior limits
        
        Parameters
        ----------
        lower_bonds : numpy1dArray
            Array containing the lower bonds of the parameter space
        upper_bonds : numpy1dArray
            Array containing the upper bonds of the parameter space
        """
        # Make sure lower / upper bonds has the same size of num_par
        if len(lower_bounds) != 4 or len(upper_bounds) != 4:
            raise ValueError("Lower and Upper bonds must be a vector with size 4")
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
    def forward_model(self, x_meas, model_par = [1, 0.7, 1.8, 4]):
        """ Forward model for pressure reconstruction at array (single freq)

        Parameters
        ----------
        model_par : TYPE, optional
            DESCRIPTION. The default is [1, 0.7, 1.8, 4].
        freq_idx : int
            Frequency index
        Returns
        -------
        prediction
        """
        h_mtx = self.build_hmtx_p((self.receivers.coord.shape[0], 2),
                                  self.current_k0, self.r1, self.r2)
        source = model_par[0] + 1j*model_par[1]
        image_source = model_par[2] + 1j*model_par[3]
        s_guess = np.array([source, image_source])
        # pressure reconstruction
        p_pred = h_mtx @ s_guess  # total pressure
        return p_pred
        
    def pk_bayesian(self, n_live = 250, max_iter = 10000, 
                    max_up_attempts = 100, seed = 0):
        
        self.ba_list = []
        
        bar = tqdm(total=len(self.controls.k0),
                   desc='Calculating Bayesian inversion (2 monopoles)...', ascii=False)
        
        # Freq loop
        for jf, self.current_k0 in enumerate(self.controls.k0):
            ba = BayesianSampler(measured_coords = self.receivers.coord, 
                                  measured_data = self.pres_s[:, jf],
                                  parameters_names = self.parameters_names,
                                  num_model_par = 4)
            ba.set_model_fun(model_fun = self.forward_model)
            ba.set_uniform_prior_limits(lower_bounds = self.lower_bounds, 
                                        upper_bounds = self.upper_bounds)
            ba.nested_sampling(n_live = n_live, max_iter = max_iter, 
                               max_up_attempts = max_iter, seed = seed)
            ba.compute_statistics()
            self.ba_list.append(ba)
            bar.update(1)
        bar.close()
    
    def ref_coeff(self,):
        vp = np.zeros(len(self.controls.k0), dtype = complex)
        alpha_vp = np.zeros(len(self.controls.k0))
        # Freq loop
        for jf, k0 in enumerate(self.controls.k0):
            mean_vals = self.ba_list[jf].mean_values
            source = mean_vals[0] + 1j*mean_vals[1]
            image_source = mean_vals[2] + 1j*mean_vals[3]
            vp[jf] = image_source/source
            alpha_vp[jf] = 1 - np.abs(vp[jf])**2
        return alpha_vp