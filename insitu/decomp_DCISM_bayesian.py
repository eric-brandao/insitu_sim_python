# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 2025

@author: Eric Brandão
"""

import numpy as np
from tqdm import tqdm
from receivers import Receiver
from sources import Source
from directDCISM import dDCISM
import matplotlib.pyplot as plt
#from sklearn.linear_model import Ridge
#from lcurve_functions_EU import csvd, l_curve, tikhonov
import lcurve_functions as lc
import utils_insitu as ut_is
# from decomp_quad_v2 import Decomposition_QDT
from baysian_sampling import BayesianSampler

class DCISM_Bayesian(object):
    """ Bayesian Decomposition of the sound field using the dDCISM.

    The class has methods to perform sound field decomposition into a set of 
    source and complex image source componensts. It is a Bayesian implementation
    that uses the dDCISM as a forward model (as implemented by M. Eser)
    """
    def __init__(self, p_mtx=None, controls=None, air = None, 
                 receivers = None, source = None):
        """

        Parameters
        ----------
        p_mtx : (N_rec x N_freq) numpy array
            A matrix containing the complex amplitudes of all the receivers
            Each column is a set of sound pressure at all receivers for a frequency.
        controls : object (AlgControls)
            Controls of the decomposition (frequency spam)
        air : object (AirProperties)
            air properties
        receivers : object (Receivers)
            Receiver properties.
        source : object (Receivers)
            Source properties

        The objects are stored as attributes in the class (easier to retrieve).
        """
        
        
        self.pres_s = p_mtx
        self.controls = controls
        self.air = air
        self.receivers = receivers
        self.source = source
        # self.parameters_names = ["Re(s)", "Im(s)", "Re(is)", "Im(is)"]
        # Get receiver data (frequency independent)
        # self.r, self.zr, self.r1, self.r2 = self.get_rec_parameters(self.receivers)
        
    def dict_forward_models(self,):
        """ Builds a dict of possible forward models and print it on screen.
        """
        available_models = {0:
                            {"name": "Locally reacting sample",
                             "num_model_par": 4,
                             "parameters_names" : [r"$Re\{\beta\}$", r"$Im\{\beta\}$", 
                                                   r"$|Q|$", r"$\angle Q$"]},
                            1:
                            {"name": "NLR layer with known thickness",
                             "num_model_par": 6,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$", 
                                                   r"$|Q|$", r"$\angle Q$"]},
                            2:
                            {"name": "NLR layer with uncertain thickness",
                             "num_model_par": 7,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$",
                                                   r"$t_p$", r"$|Q|$", r"$\angle Q$"]},
                            3:
                            {"name": "NLR semi-infinite layer",
                             "num_model_par": 6,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$", 
                                                   r"$|Q|$", r"$\angle Q$"]}
                            }
        return available_models
    
    def show_available_models(self,):
        """ Display (terminal) the available models.
        """
        available_models  = self.dict_forward_models()
        for jm in range(len(available_models)):
            print("Model {}:".format(jm))
            print("\tName: {}".format(available_models[jm]['name']))
            print("\tNumber of model parameters: {}".format(available_models[jm]['num_model_par']))
            print("\tParameters names: {}".format(available_models[jm]['parameters_names']))
            
    def choose_forward_model(self, chosen_model = 1):
        """ Chooses a model from the available models.
        """
        self.chosen_model = chosen_model
        available_models  = self.dict_forward_models()
        self.chosen_model_name = available_models[self.chosen_model]['name']
        self.num_model_par = available_models[self.chosen_model]['num_model_par']
        self.parameters_names = available_models[self.chosen_model]['parameters_names']
        
    def show_chosen_model_parameters(self,):
        """ Display (terminal) the chosen model.
        """
        available_models  = self.dict_forward_models()
        print("Model {}:".format(self.chosen_model))
        print("\tName: {}".format(available_models[self.chosen_model]['name']))
        print("\tNumber of model parameters: {}".format(available_models[self.chosen_model]['num_model_par']))
        print("\tParameters names: {}".format(available_models[self.chosen_model]['parameters_names']))
    
    def set_sample_thickness(self, t_p = 0.05):
        """ Set the sample thickness for model 1 (layer with known thickness)
        
        Parameters
        ----------
        t_p : float
            Material layer thickness
        """
        self.t_p = t_p
    
    def set_prior_limits(self, lower_bounds = [3.00, -56.00, 1.20, -64.00, 1e-5, -np.pi], 
                         upper_bounds = [189.00, -1.00, 17.00, -0.15, 1e-3, np.pi]):
        """ Set the uniform prior limits.
        
        The specified range on k_p and \rho_p was set with simulation. 
        
        Parameters
        ----------
        lower_bonds : numpy1dArray
            Array containing the lower bonds of the parameter space
        upper_bonds : numpy1dArray
            Array containing the upper bonds of the parameter space
        """
        # Make sure lower / upper bonds has the same size of num_par
        if len(lower_bounds) != self.num_model_par or len(upper_bounds) != self.num_model_par:
            raise ValueError("Lower and Upper bonds must be a vector with size {}".format(self.num_model_par))
        self.lower_bounds = lower_bounds      
        self.upper_bounds = upper_bounds
        # Make sure that the lower bound on Re{rho_p} is not smaller than rho_0
        if self.chosen_model != 0 and self.lower_bounds[2]<self.air.rho0:
            self.lower_bounds[2] = self.air.rho0
            
    def setup_dDCISM(self,  T0 = 7.5, dt = 0.1, tol = 1e-6, gamma=1.0):
        """ Initialization of object:
            
        Parameters
        -------------
        T0 : float
            End value of integration path
        dt : float
            Sample rate at which reflection coeffcient is sampled
        tol : float
            Truncation threshold for SVD in MP
        gamma : float
            Scaling factor for extended range analysis
        """
        self.dDCISMsf = dDCISM(air = self.air, controls = self.controls, 
                               source = self.source, receivers = self.receivers,
                               T0 = T0, dt = dt, tol = tol, gamma = gamma)
            
    def forward_model_1(self, x_meas, model_par = [3.00, -56.00, 1.20, -64.00, 1e-5, -np.pi]):
        """ Forward model for pressure reconstruction at array (single freq)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : TYPE, optional
            DESCRIPTION. The default is [3.00, -56.00, 1.20, -64.00, 1e-5, -np.pi].
        freq_idx : int
            Frequency index
        Returns
        -------
        prediction
        """
        # Source strength
        vol_velocity = model_par[4]*np.exp(1j*model_par[5])
        source_strength = 1j*self.air.rho0*self.controls.c0*self.current_k0*vol_velocity
        # complex wave-number
        k_p = model_par[0] + 1j*model_par[1]
        # complex density
        rho_p = model_par[2] + 1j*model_par[3]
        green_fun = self.dDCISMsf.predict_p(k = self.current_k0, k_p = k_p, rho_p = rho_p, 
                                            t_p = self.t_p)
        p_pred = source_strength * green_fun
        return p_pred
    
    def pk_bayesian(self, n_live = 250, max_iter = 10000, 
                    max_up_attempts = 100, seed = 0):
        
        self.ba_list = []
        
        bar = tqdm(total=len(self.controls.k0),
                   desc='Calculating Bayesian inversion (dDCISM)...', ascii=False)
        
        # Freq loop
        for jf, self.current_k0 in enumerate(self.controls.k0):
            ba = BayesianSampler(measured_coords = self.receivers.coord, 
                                  measured_data = self.pres_s[:, jf],
                                  parameters_names = self.parameters_names,
                                  num_model_par = self.num_model_par)
            ba.set_model_fun(model_fun = self.forward_model_1)
            ba.set_uniform_prior_limits(lower_bounds = self.lower_bounds, 
                                        upper_bounds = self.upper_bounds)
            ba.nested_sampling(n_live = n_live, max_iter = max_iter, 
                               max_up_attempts = max_iter, seed = seed)
            ba.compute_statistics()
            self.ba_list.append(ba)
            bar.update(1)
        bar.close()