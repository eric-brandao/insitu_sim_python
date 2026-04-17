# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 2025

@author: Eric Brandão
"""

import numpy as np
from controlsair import cart2sph
from receivers import Receiver
from tqdm import tqdm
from sources import Source
import material as mat
from material import PorousAbsorber, kp_rhop_range_study
from material import get_min_kp_rhop, get_max_kp_rhop, get_min_beta, get_max_beta
from directDCISM import dDCISM
import matplotlib.pyplot as plt
#from sklearn.linear_model import Ridge
#from lcurve_functions_EU import csvd, l_curve, tikhonov
import lcurve_functions as lc
from IPython.display import clear_output

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
                 receivers = None, source = None, sampling_scheme = 'single ellipsoid',
                 enlargement_factor = 1.25):
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
        self.sampling_scheme = sampling_scheme
        self.enlargement_factor = enlargement_factor
        # self.set_reference_sensor(ref_sens = 0)
        # self.parameters_names = ["Re(s)", "Im(s)", "Re(is)", "Im(is)"]
        # Get receiver data (frequency independent)
        # self.r, self.zr, self.r1, self.r2 = self.get_rec_parameters(self.receivers)
        
    def dict_forward_models(self,):
        """ Builds a dict of possible forward models and print it on screen.
        """
        available_models = {10:
                            {"name": "Locally reacting sample - H(f)",
                             "num_model_par": 2,
                             "parameters_names" : [r"$Re\{\beta\}$", r"$Im\{\beta\}$"],
                             'known_thickness': True,
                             'known_source': True,
                             'TF' : True},
                            11:
                             {"name": "LR sample (unknown source) - H(f)",
                              "num_model_par": 3,
                              "parameters_names" : [r"$Re\{\beta\}$", r"$Im\{\beta\}$", "r$z_s$"],
                              'known_thickness': True,
                              'known_source': False,
                              'TF' : True},
                            12:
                            {"name": "LR sample - P(f)",
                             "num_model_par": 4,
                             "parameters_names" : [r"$Re\{\beta\}$", r"$Im\{\beta\}$", 
                                                   r"$|S|$", r"$\angle S$"],
                             'known_thickness': True,
                             'known_source': True,
                             'TF' : False},
                            
                            20:
                            {"name": "NLR semi-infinite layer - H(f)",
                             "num_model_par": 4,
                             "parameters_names" : [r"$Re\{k_p/k_0\}$", r"$Im\{k_p/k_0\}$",
                                                   r"$Re\{\rho_p/\rho_0\}$", 
                                                   r"$Im\{\rho_p/\rho_0\}$"],
                             'known_thickness': True,
                             'known_source': True,
                             'TF' : True},
                            21:
                            {"name": "NLR semi-infinite layer - P(f)",
                             "num_model_par": 6,
                             "parameters_names" : [r"$Re\{k_p/k_0\}$", r"$Im\{k_p/k_0\}$",
                                                   r"$Re\{\rho_p/\rho_0\}$", 
                                                   r"$Im\{\rho_p/\rho_0\}$",
                                                   r"$|S|$", r"$\angle S$"],
                             'known_thickness': True,
                             'known_source': True,
                             'TF' : False},
                            
                            40:
                            {"name": "NLR single layer with known thickness/source - H(f)",
                             "num_model_par": 4,
                             "parameters_names" : [r"$Re\{k_p/k_0\}$", 
                                                   r"$Im\{k_p/k_0\}$",
                                                   r"$Re\{\rho_p/\rho_0\}$", 
                                                   r"$Im\{\rho_p/\rho_0\}$"],
                             'known_thickness': True,
                             'known_source': True,
                             'TF' : True},
                            41:
                            {"name": "NLR single layer with unknown thickness - H(f)",
                             "num_model_par": 5,
                             "parameters_names" : [r"$Re\{k_p/k_0\}$", 
                                                   r"$Im\{k_p/k_0\}$",
                                                   r"$Re\{\rho_p/\rho_0\}$", 
                                                   r"$Im\{\rho_p/\rho_0\}$", 
                                                   r"$t_p$"],
                             'known_thickness': False,
                             'known_source': True,
                             'TF' : True},
                            42:
                            {"name": "NLR single layer with unknown source - H(f)",
                             "num_model_par": 5,
                             "parameters_names" : [r"$Re\{k_p/k_0\}$", 
                                                   r"$Im\{k_p/k_0\}$",
                                                   r"$Re\{\rho_p/\rho_0\}$", 
                                                   r"$Im\{\rho_p/\rho_0\}$", 
                                                   r"$z_s$"],
                             'known_thickness': True,
                             'known_source': False,
                             'TF' : True},
                            43:
                            {"name": "NLR single layer with unknown thickness and source - H(f)",
                             "num_model_par": 6,
                             "parameters_names" : [r"$Re\{k_p/k_0\}$", 
                                                   r"$Im\{k_p/k_0\}$",
                                                   r"$Re\{\rho_p/\rho_0\}$", 
                                                   r"$Im\{\rho_p/\rho_0\}$", 
                                                   r"$t_p$", 
                                                   r"$z_s$"],
                             'known_thickness': False,
                             'known_source': False,
                             'TF' : True},
                            44:
                            {"name": "NLR single layer with known thickness - P(f)",
                             "num_model_par": 6,
                             "parameters_names" : [r"$Re\{k_p/k_0\}$", 
                                                   r"$Im\{k_p/k_0\}$",
                                                   r"$Re\{\rho_p/\rho_0\}$", 
                                                   r"$Im\{\rho_p/\rho_0\}$",
                                                   r"$|S|$", r"$\angle S$"],
                             'known_thickness': True,
                             'known_source': True,
                             'TF' : False},
                            45:
                            {"name": "NLR single layer with unknown thickness - P(f)",
                             "num_model_par": 7,
                             "parameters_names" : [r"$Re\{k_p/k_0\}$", 
                                                   r"$Im\{k_p/k_0\}$",
                                                   r"$Re\{\rho_p/\rho_0\}$", 
                                                   r"$Im\{\rho_p/\rho_0\}$",
                                                   r"$|S|$", 
                                                   r"$\angle S$",  
                                                   r"$t_p$"],
                             'known_thickness': False,
                             'known_source': True,
                             'TF' : False},
                            50:
                            {"name": "PW-NLR single layer with known thickness/source - H(f)",
                             "num_model_par": 4,
                             "parameters_names" : [r"$Re\{k_p/k_0\}$", 
                                                   r"$Im\{k_p/k_0\}$",
                                                   r"$Re\{\rho_p/\rho_0\}$", 
                                                   r"$Im\{\rho_p/\rho_0\}$"],
                             'known_thickness': True,
                             'known_source': True,
                             'TF' : True}
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
            
    def choose_forward_model(self, chosen_model = 10):
        """ Chooses a model from the available models.
        """
        self.chosen_model = chosen_model
        available_models  = self.dict_forward_models()
        self.chosen_model_dict = available_models[self.chosen_model]
        self.chosen_model_name = available_models[self.chosen_model]['name']
        self.num_model_par = available_models[self.chosen_model]['num_model_par']
        self.parameters_names = available_models[self.chosen_model]['parameters_names']
        self.get_model_fun()
        
    def get_model_fun(self,):
        """ Gets callable model function
        """
        if self.chosen_model == 10:
            self.model_fun = self.forward_model_10
        elif self.chosen_model == 40:
            self.model_fun = self.forward_model_40
        elif self.chosen_model == 41:
            self.model_fun = self.forward_model_41
        elif self.chosen_model == 42:
            self.model_fun = self.forward_model_42
        elif self.chosen_model == 43:
            self.model_fun = self.forward_model_43
        elif self.chosen_model == 44:
            self.model_fun = self.forward_model_44
        elif self.chosen_model == 45:
            self.model_fun = self.forward_model_45
        elif self.chosen_model == 50:
            self.model_fun = self.forward_model_50
        
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
        
    def set_thickness_prior_limits(self, tp_bounds = [0.001, 1]):
        self.tp_bounds = tp_bounds
        if self.tp_bounds[0] < 0.001: #1mm thickness limit
            self.tp_bounds[0] = 0.001
        if self.tp_bounds[1] > 1.0: #1.0 m thickness limit
            self.tp_bounds[1] = 1.0
            
    def set_source_height_prior_limits(self, zs_bounds = [0.25, 0.35]):
        self.zs_bounds = zs_bounds
        if self.zs_bounds[0] < 0.05: #5cm limit
            self.zs_bounds[0] = 0.05
            
    def set_source_stength_prior_limits(self, mag_s_bounds = [0.001, 2]):
        self.mag_s_bounds = mag_s_bounds
        if self.mag_s_bounds[0] < 0.001: #1mm thickness limit
            self.mag_s_bounds[0] = 0.001
        self.pha_s_bounds = [-np.pi, np.pi]
    
    def set_prior_limits(self, 
                         lower_bounds = [1.01, -5.00, 1.00, -5.00, 0.04, 0.5, -np.pi], 
                         upper_bounds = [3.00, -0.05, 1.20, -0.40, 0.06, 2.00, np.pi]):
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
        
        lower_bounds, upper_bounds = self.set_physical_limits(lower_bounds, upper_bounds)
        return lower_bounds, upper_bounds
        
    def set_prior_limits_from_study(self, jf = 0):
        """ Set the uniform prior limits.
        
        The specified range on k_p and \rho_p was set with simulation. 
        
        Parameters
        ----------
        
        """
        # Apply physical filter
        lower_bounds, upper_bounds = self.set_physical_limits(lower_bounds = self.lb_mtx[:, jf], 
                                                              upper_bounds = self.ub_mtx[:, jf])
        self.lb_mtx[:, jf] = np.copy(lower_bounds)
        self.ub_mtx[:, jf] = np.copy(upper_bounds)
        # Add source strength
        if not self.chosen_model_dict['TF']:
            lower_bounds = np.append(lower_bounds, self.mag_s_bounds[0])
            upper_bounds = np.append(upper_bounds, self.mag_s_bounds[1])
            lower_bounds = np.append(lower_bounds, self.pha_s_bounds[0])
            upper_bounds = np.append(upper_bounds, self.pha_s_bounds[1])
        # Add thickness
        if not self.chosen_model_dict['known_thickness']:
            lower_bounds = np.append(lower_bounds, self.tp_bounds[0])
            upper_bounds = np.append(upper_bounds, self.tp_bounds[1])
            if not self.chosen_model_dict['known_source']:
                lower_bounds = np.append(lower_bounds, self.zs_bounds[0])
                upper_bounds = np.append(upper_bounds, self.zs_bounds[1])
        else:
            if not self.chosen_model_dict['known_source']:
                lower_bounds = np.append(lower_bounds, self.zs_bounds[0])
                upper_bounds = np.append(upper_bounds, self.zs_bounds[1])
                
        return lower_bounds, upper_bounds
        
    def set_physical_limits(self, lower_bounds, upper_bounds):
        """ Set sensible physical limits to wave-number and density
        """
        if self.chosen_model != 10 and self.chosen_model != 11:
            # Re{k_p} constraint
            # if lower_bounds[0] < 1.00:
            #     lower_bounds[0] = 1.00
            lower_bounds[0] = 1.00
            # Im{k_p} constraint
            if upper_bounds[1] > -0.01:
                upper_bounds[1] = -0.01
            # Re{rho_p} constraint    
            # if lower_bounds[2] < 0.9: # 1.18/1.3
            #     lower_bounds[2] = 0.9
            lower_bounds[2] = 0.9
            # Im{rho_p} constraint    
            if upper_bounds[3] > -0.01: # 1.18/1.3
                upper_bounds[3] = -0.01
        else:
            if lower_bounds[0] < 0.001: # Do not allow for real part lower than zero
                lower_bounds[0] = 0.001
        return lower_bounds, upper_bounds
    
    def check_sample_physical_limits(self, sample_vec):
        """ Set sensible physical limits to wave-number and density
        """
        if self.chosen_model != 10 and self.chosen_model != 11:
            # Re{k_p} constraint
            if sample_vec[0] < 1.00:
                sample_vec[0] = 1.00
            # Im{k_p} constraint
            if sample_vec[1] > -0.01:
                sample_vec[1] = -0.01
            # Re{rho_p} constraint    
            if sample_vec[2] < 0.9: # 1.18/1.3
                sample_vec[2] = 0.9
            # Im{rho_p} constraint    
            if sample_vec[3] > -0.01: # 1.18/1.3
                sample_vec[3] = -0.01
        else:
            if sample_vec[0] < 0.001: # Do not allow for real part lower than zero
                sample_vec[0] = 0.001
        return sample_vec
    
    def wide_prior_range(self, widening_factor = 2):
        """ Wide your prior by a given factor
        """
        assert widening_factor >= 0
        delta = np.array(self.upper_bounds) - np.array(self.lower_bounds)
        delta2sum = (delta/2)*((widening_factor-1))#-(delta/4)
        self.lower_bounds -= delta2sum
        self.upper_bounds += delta2sum
        if self.chosen_model != 10 or self.chosen_model != 11:
            self.set_physical_limits()
    
    def set_nested_sampling_parameters(self, n_live = 50, max_iter = 2000, 
                                       max_up_attempts = 50, seed = 0, dlogz = 0.1,
                                       ci_percent = 95):
        """ Setup nested sampling parameters. Valid for all runs
        
        Parameters
        ----------
        n_live : int
            The number of live points in your initial and final live population.
            It will contain your initial population, and be updated. As the iterations
            go on, the likelihood of such set of points will increase until termination.
        max_iter : int
            The maximum number of iterations allowed.
        max_up_attempts : int
            number of attempts to update the parameter set of lowest log-likelihood
            to a parameter set with higher likelihood.
        seed : int
            seed for random number generator.
        dlogz : float
            minimum uncertainty in logZ (for termination)
        ci_percent : float
            Confidence interval for uncertainty estimation. You can specify it for running,
            but recompute it easily later on.
        """
        self.n_live = n_live
        self.max_iter = max_iter
        self.max_up_attempts = max_up_attempts
        self.seed = seed
        self.dlogz = dlogz
        self.ci_percent = ci_percent
    
    def set_reference_sensor(self, ref_sens = 0):
        """ Set the reference sensor index, new receiver array and measured data.
        
        Parameters
        ----------
        ref_sens : int
            reference sensor index
        """
        # Set ref sensor
        self.ref_sens = ref_sens
        # Set new receiver array
        new_receivers = Receiver()
        new_receivers.coord = np.delete(self.receivers.coord, self.ref_sens, axis = 0)
        self.receivers = new_receivers 
        # Set new measurement data
        self.pres_s = self.pres_s / self.pres_s[self.ref_sens,:]
        self.pres_s = np.delete(self.pres_s, self.ref_sens, axis = 0)
        
    def set_mic_pairs(self,):
        """ Get the indexes of mic-pair transfer functions
        """
        # Collect mic-pair transfer funtions
        self.id_z_list = self.receivers.get_micpair_indices()
        self.tf_array = Receiver()
        self.tf_array.coord = self.receivers.coord[self.id_z_list[0],:]
        # Change measured data to transfer function
        p_exp_line_1 = self.pres_s[self.id_z_list[0],:]
        p_exp_line_2 = self.pres_s[self.id_z_list[1],:]
        self.pres_s = p_exp_line_1/p_exp_line_2
            
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
    
    def forward_model_10(self, x_meas, 
                        model_par = [0.5, 1.00]):
        """ Forward model for forward prediction
        
        Valid for Model 11 - LR single layer - H(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{beta}, Im{beta}].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex surface admittance
        beta = model_par[0] + 1j*model_par[1]       
        green_fun = self.dDCISMsf.predict_p_lr(k = self.current_k0, 
                                               beta = beta)
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        pred = p_exp_line_1/p_exp_line_2
        return pred
    
    def forward_model_40(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50]):
        """ Forward model for forward prediction
        
        Valid for Model 40 - NLR single layer with known thickness - H(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = self.current_k0*(model_par[0] + 1j*model_par[1])
        # complex density
        rho_p = self.air.rho0*(model_par[2] + 1j*model_par[3])
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, 
                                                      rho_p = rho_p, 
                                                      t_p = self.t_p)
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        pred = p_exp_line_1/p_exp_line_2
        return pred
    
    def forward_model_41(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50, 0.05]):
        """ Forward model for forward prediction
        
        Valid for Model 41 - NLR single layer with unknown thickness - H(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}, t_p].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = self.current_k0*(model_par[0] + 1j*model_par[1])
        # complex density
        rho_p = self.air.rho0*(model_par[2] + 1j*model_par[3])
        # unknown thickness
        t_p = model_par[4]
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, 
                                                      rho_p = rho_p, 
                                                      t_p = t_p)
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        pred = p_exp_line_1/p_exp_line_2
        return pred
    
    def forward_model_42(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50, 0.30]):
        """ Forward model for forward prediction
        
        Valid for Model 42 - NLR single layer with unknown source - H(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}, t_p, z_s].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = self.current_k0*(model_par[0] + 1j*model_par[1])
        # complex density
        rho_p = self.air.rho0*(model_par[2] + 1j*model_par[3])
        # unknown thickness
        self.dDCISMsf.hs = model_par[4]
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, 
                                                      rho_p = rho_p, 
                                                      t_p = self.t_p)
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        pred = p_exp_line_1/p_exp_line_2
        return pred
    
    def forward_model_43(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50, 0.05, 0.30]):
        """ Forward model for forward prediction
        
        Valid for Model 42 - NLR single layer with unknown thickness and source - H(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}, t_p, z_s].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = self.current_k0*(model_par[0] + 1j*model_par[1])
        # complex density
        rho_p = self.air.rho0*(model_par[2] + 1j*model_par[3])
        # unknown thickness / source
        t_p = model_par[4]
        self.dDCISMsf.hs = model_par[5]
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, 
                                                      rho_p = rho_p, 
                                                      t_p = t_p)
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        pred = p_exp_line_1/p_exp_line_2
        return pred
    
    def forward_model_44(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50, 1.00, -np.pi]):
        """ Forward model for forward prediction
        
        Valid for Model 43 - NLR single layer with known thickness - P(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}, |S|, \angle(S)].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = model_par[0] + 1j*model_par[1]
        # complex density
        rho_p = model_par[2] + 1j*model_par[3]
        # Source strength
        source_strength = model_par[4]*np.exp(1j*model_par[5])
        
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, rho_p = rho_p, 
                                                      t_p = self.t_p)
        pred = source_strength * green_fun
        return pred
    
    def forward_model_45(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50, 0.05, 1.00, -np.pi]):
        """ Forward model for forward prediction
        
        Valid for Model 44 - NLR single layer with unknown thickness - P(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}, t_p, |S|, \angle(S)].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # complex wave-number
        k_p = model_par[0] + 1j*model_par[1]
        # complex density
        rho_p = model_par[2] + 1j*model_par[3]
        # unknown thickness
        t_p = model_par[4]
        # Source strength
        source_strength = model_par[4]*np.exp(1j*model_par[5])
        
        green_fun = self.dDCISMsf.predict_p_nlr_layer(k = self.current_k0, 
                                                      k_p = k_p, rho_p = rho_p, 
                                                      t_p = t_p)
        pred = source_strength * green_fun
        return pred
    
    def forward_model_50(self, x_meas, 
                        model_par = [1.50, -1.00, 1.20, -2.50]):
        """ Forward model for forward prediction
        
        Valid for Model 4 - NLR single layer with known thickness - H(f)

        Parameters
        ----------
        x_meas : numpyndArray
            Measurement coordinates
        model_par : list
            Guessing values for [Re{k_p}, Im{k_p}, Re{\rho_p}, Im{\rho_p}].
        Returns
        -------
        pred : numpy1dArray
            Complex transfer function between microphone pairs
        """
        # Angular freq
        omega = self.air.c0 * self.current_k0
        # complex wave-number
        k_p = self.current_k0*(model_par[0] + 1j*model_par[1])
        # complex density
        rho_p = self.air.rho0*(model_par[2] + 1j*model_par[3])
        # Get source wave-number as a plane wave
        src_unit_vec = self.source.coord[0,:]/np.linalg.norm(self.source.coord[0,:])
        k_ref = self.current_k0 * src_unit_vec
        k_inc = np.copy(k_ref)
        k_inc[2] = -k_inc[2]
        _, theta, _ = cart2sph(k_ref[0], k_ref[1], k_ref[2])
        theta = np.pi/2-theta 
        
        # Material properties
        Zp = rho_p*omega/k_p
        theta_t = np.arcsin(self.current_k0*np.sin(theta)/k_p)
        Zs = -1j*(Zp/np.cos(theta_t))*(1/np.tan(k_p*np.cos(theta_t)*self.t_p))
        Vp = (Zs*np.cos(theta)-self.air.rho0*self.air.c0)/\
            (Zs*np.cos(theta)+self.air.rho0*self.air.c0)        
        # green fun - plane wave
        green_fun = np.exp(-1j*self.receivers.coord @ k_inc)+\
            Vp*np.exp(-1j*self.receivers.coord @ k_ref)    
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        pred = p_exp_line_1/p_exp_line_2
        return pred
    
    def nested_sampling_single_freq(self, lower_bounds, upper_bounds, 
                                    jf = 0):
        """ Run single freq inference
        
        Parameters
        ----------
        lower_bounds : numpy 1dArray
            lower bounds for that frequency. If None, then we choose the lower bounds from
            the studied search range.
        jf : int
            Freq index to run
        """
        self.current_k0 = self.controls.k0[jf]
        ba = BayesianSampler(measured_coords = self.receivers.coord[self.id_z_list[0],:], 
                              measured_data = self.pres_s[:, jf],
                              parameters_names = self.parameters_names,
                              num_model_par = self.num_model_par,
                              sampling_scheme = self.sampling_scheme,
                              enlargement_factor = self.enlargement_factor)
        ba.set_model_fun(model_fun = self.model_fun)
        ba.set_uniform_prior_limits(lower_bounds = lower_bounds, 
                                    upper_bounds = upper_bounds)
        ba.nested_sampling(n_live = self.n_live, max_iter = self.max_iter, 
                           max_up_attempts = self.max_up_attempts, seed = self.seed,
                           dlogz = self.dlogz)
        # # ba.ultranested_sampling(n_live = self.n_live, max_iter = self.max_iter)
        # # ba.ultranested_sampling_react(n_live = self.n_live, max_iter = self.max_iter)
        ba.compute_statistics(ci_percent = self.ci_percent)
        return ba
    
    def nested_sampling_spk(self):
        """ Run inference for all frequency bins
        
        Parameters
        ----------
        """
        # empty list with all Bayesian inference objects.
        self.ba_list = []
        self.logZ_spk = np.zeros(len(self.controls.k0))
        self.logZ_err_spk = np.zeros(len(self.controls.k0))
        # Freq loop
        for jf, self.current_k0 in enumerate(self.controls.k0):
            # Message
            print("NS run for {:.2f} [Hz] ({} of {} bins)".format(self.controls.freq[jf],
                                                              jf+1, len(self.controls.k0)))
            
            # get bounds from study
            lb, ub = self.set_prior_limits_from_study(jf = jf)
            # run Bayesian inference
            ba = self.nested_sampling_single_freq(lb, ub, jf = jf)
            self.ba_list.append(ba)
            self.logZ_spk[jf] = ba.logZ
            self.logZ_err_spk[jf] = ba.logZ_err
            clear_output()
        print("Inference frequency loop finished!")
        
    def nested_sampling_spk2(self, freqs_init = [700, 1000, 1500], 
                             resist_range = 10000, resist_lb = 1000,
                             res_factor = 2):
        """ Run inference for all frequency bins
        
        Parameters
        ----------
        freqs_init : list
            list of frequencies for initial estimation
        """
        resist_freq = np.zeros(len(freqs_init))
        for jf, f in enumerate(freqs_init):
            # Message
            print("NS init run for {:.2f} [Hz] ({} of {} bins)".format(f, jf+1, len(freqs_init)))
            # freq index to run
            idf = ut_is.find_freq_index(freq_vec = self.controls.freq, freq_target = f)
            # get bounds from study
            lb, ub = self.set_prior_limits_from_study(jf = idf)
            # run Bayesian inference
            ba = self.nested_sampling_single_freq(lb, ub, jf = idf)
            # Compute flow resistivity from each freq
            rhop_mean_im = self.air.rho0*ba.mean_values[3]
            resist_freq[jf] = -self.controls.w[idf] * rhop_mean_im
            clear_output()
        print("Initial inference frequency loop finished!")
        # mean flow - resistivity across spk
        resist_mean = np.mean(resist_freq)
        print(r"Mean flow resistivity for init run: {} [Nsm$^-4$]".format(resist_mean))
        # Min/Max values before material study
        resist_min = resist_mean - resist_range
        if resist_min < resist_lb:
            resist_min = resist_lb
        resist_max = resist_mean + resist_range
        # Re-run material study
        self.kp_rhop_range_miki(resist = [resist_min, resist_max],
                                n_samples = 20000) #        self.kp_rhop_range_miki(resist = ,

        # self.kp_rhop_range(resist = [(1/res_factor)*resist_mean, res_factor*resist_mean], 
        #                    n_samples = 20000,
        #                    phi = [0.8, 0.99], alpha_inf = [1.0, 1.5], 
        #                    Lam = [100e-6, 600e-6], Lamlfac = [1.01, 2.0])
     
        # Run whole spk
        self.nested_sampling_spk()            

    def nested_sampling_spk_seqg(self, start_freq = 1000, fac = 5):
        """ Sequential nested sampling with gauss prior
        
        """
        # empty list with all Bayesian inference objects.
        self.ba_list = [None]*len(self.controls.k0)
        self.logZ_spk = np.zeros(len(self.controls.k0))
        self.logZ_err_spk = np.zeros(len(self.controls.k0))
        # starting index
        start_id = ut_is.find_freq_index(freq_vec = self.controls.freq, 
                                         freq_target = start_freq)
        id_vec = np.arange(0, len(self.controls.freq))
        id_vec_shifted = np.concatenate((id_vec[start_id:], id_vec[:start_id][::-1]))
        freq_vec_shifted = self.controls.freq[id_vec_shifted]
        k0_vec_shifted = self.controls.k0[id_vec_shifted]
        
        # Initial phase
        lb, ub = self.set_prior_limits_from_study(jf = start_id)
        delta_ulb = np.abs(ub-lb)
        
        print("NS run (INIT) for {:.2f} [Hz] ({} of {} bins)".format(freq_vec_shifted[0],
                                                                     1, len(freq_vec_shifted)))
        ba = self.nested_sampling_single_freq(lb, ub, jf = id_vec_shifted[0])
        # Put it in the correct place at list
        self.ba_list[id_vec_shifted[0]] = ba
        self.logZ_spk[id_vec_shifted[0]] = ba.logZ
        self.logZ_err_spk[id_vec_shifted[0]] = ba.logZ_err
        # delta_ulb = np.abs(ba.ci[1,:]-ba.ci[0,:])
        
        # Freq loop
        for jf in np.arange(1, len(freq_vec_shifted)):
            # Message
            print("NS run for {} [Hz] ({} of {} bins)".format(freq_vec_shifted[jf],
                                                              jf+1, len(freq_vec_shifted)))
            self.current_k0 = k0_vec_shifted[jf]
            mu = ba.mean_values
            sigma_mtx = np.diag(delta_ulb)
            
            ba = BayesianSampler(measured_coords = self.receivers.coord[self.id_z_list[0],:], 
                                 measured_data = self.pres_s[:, jf],
                                 parameters_names = self.parameters_names,
                                 num_model_par = self.num_model_par,
                                 sampling_scheme = self.sampling_scheme,
                                 enlargement_factor = self.enlargement_factor,
                                 uniform_prior = False)
            ba.set_model_fun(model_fun = self.model_fun)
            ba.set_gauss_prior(mu = mu, sigma_mtx = sigma_mtx)
            # ba.set_uniform_prior_limits(lower_bounds = mu-delta_ulb, 
            #                             upper_bounds = mu+delta_ulb)
            ba.nested_sampling(n_live = self.n_live, max_iter = self.max_iter, 
                               max_up_attempts = self.max_up_attempts, seed = self.seed,
                               dlogz = self.dlogz)
            # # ba.ultranested_sampling(n_live = self.n_live, max_iter = self.max_iter)
            # # ba.ultranested_sampling_react(n_live = self.n_live, max_iter = self.max_iter)
            ba.compute_statistics(ci_percent = self.ci_percent)
            self.ba_list[id_vec_shifted[jf]] = ba
            self.logZ_spk[id_vec_shifted[jf]] = ba.logZ
            self.logZ_err_spk[id_vec_shifted[jf]] = ba.logZ_err
        
        
    def nested_sampling_spk_seq(self, start_freq = 1000, fac = 5):
        """ Run inference for all frequency bins
        
        Parameters
        ----------
        """
        # lb, ub matrices
        lb_mtx = np.zeros(self.lb_mtx.shape)
        ub_mtx = np.zeros(self.ub_mtx.shape)
        
        # empty list with all Bayesian inference objects.
        self.ba_list = [None]*len(self.controls.k0)
        self.logZ_spk = np.zeros(len(self.controls.k0))
        self.logZ_err_spk = np.zeros(len(self.controls.k0))
        # starting index
        start_id = ut_is.find_freq_index(freq_vec = self.controls.freq, 
                                         freq_target = start_freq)
        id_vec = np.arange(0, len(self.controls.freq))
        id_vec_shifted = np.concatenate((id_vec[start_id:], id_vec[:start_id][::-1]))
        freq_vec_shifted = self.controls.freq[id_vec_shifted]
        k0_vec_shifted = self.controls.k0[id_vec_shifted]
        
        # Initial phase
        lb, ub = self.set_prior_limits_from_study(jf = start_id)
        print("NS run (INIT) for {:.2f} [Hz] ({} of {} bins)".format(freq_vec_shifted[0],
                                                                     1, len(freq_vec_shifted)))
        ba = self.nested_sampling_single_freq(lb, ub, jf = id_vec_shifted[0])
        # Put it in the correct place at list
        self.ba_list[id_vec_shifted[0]] = ba
        self.logZ_spk[id_vec_shifted[0]] = ba.logZ
        self.logZ_err_spk[id_vec_shifted[0]] = ba.logZ_err
        lb_mtx[:,id_vec_shifted[0]] = lb
        ub_mtx[:,id_vec_shifted[0]] = ub
        # New bounds
        # fac = 10
        # lb0 = ba.mean_values - fac*ba.std
        # ub0 = ba.mean_values + fac*ba.std
        #delta_lb_mu = ba.mean_values - lb
        #delta_ub_mu = ub - ba.mean_values
        #lb0 = ba.mean_values - delta_lb_mu
        #ub0 = ba.mean_values + delta_ub_mu
        lb0, ub0 = lb, ub
        lb0, ub0 = self.set_physical_limits(lb0, ub0)
        
        # Freq loop
        for jf in np.arange(1, len(freq_vec_shifted)):
            # Message
            print("NS run for {} [Hz] ({} of {} bins)".format(freq_vec_shifted[jf],
                                                              jf+1, len(freq_vec_shifted)))
            self.current_k0 = k0_vec_shifted[jf]
                      
            if id_vec_shifted[jf] == start_id + 1 or id_vec_shifted[jf] == start_id - 1:
                lb = np.copy(lb0)
                ub = np.copy(ub0)
            else:
                # lb = ba.mean_values - fac*ba.std
                # ub = ba.mean_values + fac*ba.std
                delta_lb_mu = ba.mean_values - lb
                delta_ub_mu = ub - ba.mean_values
                lb = ba.mean_values - delta_lb_mu
                ub = ba.mean_values + delta_ub_mu
                lb, ub = self.set_physical_limits(lb, ub)
                
            ba = self.nested_sampling_single_freq(lb, ub, jf = id_vec_shifted[jf])
            self.ba_list[id_vec_shifted[jf]] = ba
            self.logZ_spk[id_vec_shifted[jf]] = ba.logZ
            self.logZ_err_spk[id_vec_shifted[jf]] = ba.logZ_err
            lb_mtx[:,id_vec_shifted[jf]] = lb
            ub_mtx[:,id_vec_shifted[jf]] = ub
        # return ba
        return lb_mtx, ub_mtx
                
            
        
    def nested_sampling_spk_seq0(self, start_freq = 1000):
        """ Run inference for all frequency bins
        
        Parameters
        ----------
        """
        # empty list with all Bayesian inference objects.
        self.ba_list = [None]*len(self.controls.k0)
        # current_freq = start_freq
        start_id = ut_is.find_freq_index(self.controls.freq, 
                                         freq_target = start_freq)
        # get bounds from study
        lb, ub = self.set_prior_limits_from_study(jf = start_id)
        id_vec = np.arange(0, len(self.controls.freq))
        id_vec_shifted = np.concatenate((id_vec[start_id:], id_vec[:start_id][::-1]))
        freq_vec_shifted = self.controls.freq[id_vec_shifted]
        #np.roll(self.controls.freq, -start_id)
        k0_vec_shifted = self.controls.k0[id_vec_shifted]
        self.logZ_spk = np.zeros(len(self.controls.k0))
        self.logZ_err_spk = np.zeros(len(self.controls.k0))
        # Freq loop
        for jf, self.current_k0 in enumerate(k0_vec_shifted):
            # Message
            print("NS run for {} [Hz] ({} of {} bins)".format(freq_vec_shifted[jf],
                                                              jf+1, len(freq_vec_shifted)))
            
            # get bounds from study
            # lb, ub = self.set_prior_limits_from_study(jf = id_vec_shifted[jf])
            # run Bayesian inference
            ba = self.nested_sampling_single_freq(lb, ub, jf = id_vec_shifted[jf])
            # Put it in the correct place at list
            self.ba_list[id_vec_shifted[jf]] = ba
            self.logZ_spk[id_vec_shifted[jf]] = ba.logZ
            self.logZ_err_spk[id_vec_shifted[jf]] = ba.logZ_err
            # Compute lower/upper bounds of new freq
            # ba.confidence_interval(ci_percent=99.99)
            # lb, ub = self.set_prior_limits_from_study(jf = id_vec_shifted[jf])
            lb = ba.mean_values - 20*ba.std
            ub = ba.mean_values + 20*ba.std
            lb, ub = self.set_physical_limits(lb, ub)
            # lb = np.copy(ba.ci[0,:])
            # ub = np.copy(ba.ci[1,:])
            print(np.vstack((lb,ub)))
            print(np.vstack((self.lb_mtx[:,id_vec_shifted[jf]],self.ub_mtx[:,id_vec_shifted[jf]])))
        
    def recon_nlr_1layer(self,):
        """ Reconstruct all quantities for single NLR layer vs. freq
        """
        # wave number
        self.get_kp_spk()
        # Density
        self.get_rhop_spk()
        # Characteristic impedance
        self.get_zp_spk()
            
    def get_kp(self, inf_value, k0):
        """ Get the complex wave-number inferred value for single frequency
        
        Parameters
        ----------
        inf_value : np1dArray or list
            values inferred of all parameters (Re and Im)
        k0 : float
            wave-number in air at a given frequency (omega/c0)
        
        Returns
        ----------
        kp : complex
            complex-valued wave-number
        """
        return k0*(inf_value[:,0] + 1j*inf_value[:,1])
    
    def get_kp_spk(self,):
        # Initialization
        self.kp_mean = np.zeros(len(self.controls.k0), dtype = complex)
        self.kp_ci = np.zeros((2, len(self.controls.k0)), dtype = complex)
        # Freq loop
        # Initialize bar
        bar = tqdm(total=len(self.controls.k0),
                   desc=r'Reconstruct $k_p$ spk.', ascii=False)
        for jf, k0 in enumerate(self.controls.k0):
            self.kp_mean[jf] = self.get_kp(np.array([self.ba_list[jf].mean_values]), k0)
            self.kp_ci[0, jf] = self.get_kp(np.array([self.ba_list[jf].ci[0,:]]), k0)
            self.kp_ci[1, jf] = self.get_kp(np.array([self.ba_list[jf].ci[1,:]]), k0)            
            bar.update(1)
        bar.close()
    
    def get_rhop(self, inf_value):
        """ Get the complex density inferred value for single frequency
        
        Parameters
        ----------
        inf_value : np1dArray or list
            values inferred of all parameters (Re and Im)
        Returns
        ----------
        rhop : complex
            complex-valued density
        """
        return self.air.rho0*(inf_value[:,2] + 1j*inf_value[:,3])
    
    def get_rhop_spk(self,):
        # Initialization
        self.rhop_mean = np.zeros(len(self.controls.k0), dtype = complex)
        self.rhop_ci = np.zeros((2, len(self.controls.k0)), dtype = complex)
        # Freq loop
        bar = tqdm(total=len(self.controls.k0),
                   desc=r'Reconstruct $\rho_p$ spk.', ascii=False)
        for jf, k0 in enumerate(self.controls.k0):
            self.rhop_mean[jf] = self.get_rhop(np.array([self.ba_list[jf].mean_values]))
            self.rhop_ci[0, jf] = self.get_rhop(np.array([self.ba_list[jf].ci[0,:]]))
            self.rhop_ci[1, jf] = self.get_rhop(np.array([self.ba_list[jf].ci[1,:]])) 
            bar.update(1)
        bar.close()

    def get_cp(self, kp, omega):
        """ Get the complex sound speed inferred value for single frequency
        
        Parameters
        ----------
        kp : complex
            complex-valued wave-number
        omega : float
            angular frequency
        
        Returns
        ----------
        cp : complex
            complex-valued sound speed
        """
        return omega/kp
    
    def get_Zp(self, kp, rhop, omega):
        """ Get the complex characteristic impedance inferred value for single frequency
        
        Parameters
        ----------
        kp : complex
            complex-valued wave-number
        rhop : complex
            complex-valued density
        omega : float
            angular frequency
        
        Returns
        ----------
        Zp : complex
            complex-valued characteristic impedance
        """
        return rhop*omega/kp
    
    def get_resit(self, rhop, omega):
        """ Get the flow resistivity inferred value for single frequency
        
        Parameters
        ----------
        rhop : complex
            complex-valued density
        omega : float
            angular frequency
        
        Returns
        ----------
        resist : float
            flow resistivity
        """
        return -omega*np.imag(rhop)
    
    def ba_recon(self, samples, weights):
        """ Reconstruct quantities that are functions of infferred parameters
        """
        # Get number of parameters
        num_model_par = samples.shape[1]
        parameters_names = ['a'] * num_model_par
        # Create empty Bayesian sampler and fill in with prior samples
        ba_rec = BayesianSampler(measured_coords = np.array([0]),
                                 measured_data = np.array([0]),
                                 num_model_par = num_model_par,
                                 parameters_names = parameters_names)
        ba_rec.prior_samples = samples 
        ba_rec.weights = weights
        # Compute the statistics
        ba_rec.compute_statistics(ci_percent = self.ci_percent)
        return ba_rec.mean_values, ba_rec.ci
    
    def get_zp_spk(self,):
        # Initialization
        self.zp_mean = np.zeros(len(self.controls.k0), dtype = complex)
        self.zp_ci = np.zeros((2, len(self.controls.k0)), dtype = complex)
        # Freq loop
        bar = tqdm(total=len(self.controls.k0),
                   desc=r'Reconstruct $Z_p$ spk.', ascii=False)
        for jf, k0 in enumerate(self.controls.k0):
            # Get the samples of kp and rhop
            kp_samples = self.get_kp(self.ba_list[jf].prior_samples, k0)
            rhop_samples = self.get_rhop(self.ba_list[jf].prior_samples)
            # Sample Zp
            zp_samples = self.get_Zp(kp_samples, rhop_samples, k0*self.air.c0)
            # Use Bayesian object to compute statistics
            mu, ci = self.ba_recon(np.array([np.real(zp_samples), np.imag(zp_samples)]).T, 
                                   self.ba_list[jf].weights)
            
            self.zp_mean[jf] = mu[0] + 1j*mu[1]
            self.zp_ci[0, jf] = ci[0, 0] + 1j*ci[0, 1]
            self.zp_ci[1, jf] = ci[1, 0] + 1j*ci[1, 1]
            bar.update(1)
        bar.close()
        
    def get_resist_spk(self,):
        # Initialization
        self.resist_mean = np.zeros(len(self.controls.k0))
        self.resist_ci = np.zeros((2, len(self.controls.k0)))
        # Freq loop
        bar = tqdm(total=len(self.controls.k0),
                   desc=r'Reconstruct Flow resistivity spk.', ascii=False)
        for jf, w in enumerate(self.controls.w):
            # Get the samples rhop
            rhop_samples = self.get_rhop(self.ba_list[jf].prior_samples)
            # Sample Zp
            resist_samples = self.get_resit(rhop_samples, w)
            # Use Bayesian object to compute statistics
            mu, ci = self.ba_recon(np.array([resist_samples]).T, self.ba_list[jf].weights)
            
            self.resist_mean[jf] = mu
            self.resist_ci[0, jf] = ci[0]
            self.resist_ci[1, jf] = ci[1]
            bar.update(1)
        bar.close()
    
    def get_theta_t(self, kp, k0, theta):
        """ Get the complex refraction angle inferred value for single frequency
        
        Parameters
        ----------
        kp : complex
            complex-valued wave-number
        k0 : float
            wave-number in air at a given frequency (omega/c0)
        theta : float or array
            angle of incidence in radians
        
        Returns
        ----------
        theta_t : complex
            complex-valued refraction angle
        """
        return np.arcsin((k0/kp)*np.sin(theta))
        
    def recon_vp_nlr_1layer(self, k0, theta, ba):
        """ NLR 1 layer planar reflection coefficient reconstruction (single freq/single angle)
        
        Reconstruct the planar reflection coefficient and its absorption coefficient.
        Valid for a single layer of non-locally reacting porous material above rigid-backing.
        It can reconstruct for single angle at a single frequency.  
        
        Parameters
        ----------
        k0 : float
            wave-number in air at a given frequency (omega/c0)
        theta : float
            angle of incidence in radians
        
        ba : object
            values inferred of all parameters (Re and Im)
        
        Returns
        ----------
        vp : complex
            complex-valued reflection coefficient
        alpha : floar
            absorption coefficient
        """
        # Get the samples of kp and rhop
        kp_samples = self.get_kp(ba.prior_samples[:,0:4], k0)
        rhop_samples = self.get_rhop(ba.prior_samples[:,0:4])
        # Get the samples of thickness
        # if self.num_model_par == 4 or self.chosen_model == 42:
        #     tp_samples = self.t_p*np.ones(ba.prior_samples.shape[0])
        # elif self.num_model_par == 5 or self.chosen_model == 41:
        #     tp_samples = ba.prior_samples[:,-1]
        # elif self.num_model_par == 6:
        #     tp_samples = ba.prior_samples[:,-2]
        if self.chosen_model_dict['known_thickness']:
            tp_samples = self.t_p*np.ones(ba.prior_samples.shape[0])
        else:
            if self.num_model_par == 5:
                tp_samples = ba.prior_samples[:,-1]
            else:
                tp_samples = ba.prior_samples[:,-2]
            
        
        # Angular wave-number in air
        k0z = k0*np.cos(theta)
        # Angular wave-number in the layer
        theta_t = self.get_theta_t(kp_samples, k0, theta)
        k1z = kp_samples*np.cos(theta_t)
        # reflection coefficient (samples)
        num = -1j*k0z/self.air.rho0 - (k1z/rhop_samples)*np.tan(k1z*tp_samples)
        den = -1j*k0z/self.air.rho0 + (k1z/rhop_samples)*np.tan(k1z*tp_samples)
        vp_samples = num/den
        # absorption coefficient (samples)
        alpha_samples = 1-np.abs(vp_samples)**2
        # Surface impedance (samples)
        zs_samples = self.recon_zs_nlr_1layer(self.air.rho0 * self.air.c0, 
                                              vp_samples, theta)
        
        # Use Bayesian object to compute statistics - Vp
        mu, ci = self.ba_recon(np.array([np.real(vp_samples), np.imag(vp_samples)]).T, 
                                       ba.weights)    
        vp_mean = mu[0] + 1j*mu[1]
        vp_ci = np.zeros(2, dtype = complex)
        vp_ci[0] = ci[0, 0] + 1j*ci[0, 1]
        vp_ci[1] = ci[1, 0] + 1j*ci[1, 1]
        
        # Use Bayesian object to compute statistics - alpha
        mu, ci = self.ba_recon(np.array([alpha_samples]).T, ba.weights)
        alpha_mean = mu[0]
        alpha_ci = np.zeros(2)
        alpha_ci[0] = ci[0, 0]
        alpha_ci[1] = ci[1, 0]
        
        # Use Bayesian object to compute statistics - Zs
        mu, ci = self.ba_recon(np.array([np.real(zs_samples), np.imag(zs_samples)]).T, 
                               ba.weights)
        zs_mean = mu[0] + 1j*mu[1]
        zs_ci = np.zeros(2, dtype = complex)
        zs_ci[0] = ci[0, 0] + 1j*ci[0, 1]
        zs_ci[1] = ci[1, 0] + 1j*ci[1, 1]
        
        return vp_mean, vp_ci.T, alpha_mean, alpha_ci.T, zs_mean, zs_ci.T
    
    def get_vp_nlr_spk(self, theta):
        # Initialization
        self.vp_mean = np.zeros((len(theta), len(self.controls.k0)), dtype = complex)
        self.vp_ci = np.zeros((2, len(theta), len(self.controls.k0)), dtype = complex)
        self.alpha_mean = np.zeros((len(theta), len(self.controls.k0)))        
        self.alpha_ci = np.zeros((2, len(theta), len(self.controls.k0)))
        self.zs_mean = np.zeros((len(theta), len(self.controls.k0)), dtype = complex)
        self.zs_ci = np.zeros((2, len(theta), len(self.controls.k0)), dtype = complex)
        
        # Freq loop
        bar = tqdm(total=len(self.controls.k0)*len(theta),
                   desc=r'Reconstruct $V_p$ (NLR) spk.', ascii=False)
        for jt, the in enumerate(theta):
            for jf, k0 in enumerate(self.controls.k0):
                var = self.recon_vp_nlr_1layer(k0, the, self.ba_list[jf])
                self.vp_mean[jt, jf] = var[0]
                self.vp_ci[:, jt, jf] = var[1]
                self.alpha_mean[jt, jf] = var[2]
                self.alpha_ci[:, jt, jf] = var[3]
                self.zs_mean[jt, jf] = var[4]
                self.zs_ci[:, jt, jf] = var[5]
                bar.update(1)
        bar.close()
                
    def get_vp_nlr_angle(self, jf, ba, theta):
        # Initialization
        vp_mean = np.zeros(len(theta), dtype = complex)
        vp_ci = np.zeros((2, len(theta)), dtype = complex)
        alpha_mean = np.zeros(len(theta))        
        alpha_ci = np.zeros((2, len(theta)))
        zs_mean = np.zeros(len(theta), dtype = complex)
        zs_ci = np.zeros((2, len(theta)), dtype = complex)
        
        # Freq loop
        for jt, the in enumerate(theta):
            var = self.recon_vp_nlr_1layer(self.controls.k0[jf], the, ba)
            vp_mean[jt] = var[0]
            vp_ci[:, jt] = var[1]
            alpha_mean[jt] = var[2]
            alpha_ci[:, jt] = var[3]
            zs_mean[jt] = var[4]
            zs_ci[:, jt] = var[5]
        return vp_mean, vp_ci.T, alpha_mean, alpha_ci, zs_mean, zs_ci
    
    def recon_zs_nlr_1layer(self, z0, vp, theta):
        """ NLR 1 layer surface impedance reconstruction (single freq)
        
        Reconstruct the surface impedance.
        Valid for a single layer of non-locally reacting porous material above rigid-backing.
        It can reconstruct for multiple angles at a single frequency.  
        
        Parameters
        ----------
        z0 : float
            characteristic impedance of air (rho0*c0)
        vp : complex
            complex-valued reflection coefficient
        theta : float or array
            angle of incidence in radians
        Returns
        ----------
        zs : complex
            complex-valued surface impedance
        """
        return (z0/np.cos(theta))*((1+vp)/(1-vp))
      
    def get_beta(self, inf_value):
        """ Get the complex surface admittance inferred value for single frequency
        
        Parameters
        ----------
        inf_value : np1dArray or list
            values inferred of all parameters (Re and Im)
        
        Returns
        ----------
        beta : complex
            complex-valued surface admittance
        """
        return inf_value[:,0] + 1j*inf_value[:,1]

    def recon_vp_lr(self, k0, theta, ba):
        """ LR planar reflection coefficient reconstruction (single freq)
        
        Reconstruct the planar reflection coefficient and its absorption coefficient.
        Valid for a locally reacting sample.
        It can reconstruct for multiple angles at a single frequency.  
        
        Parameters
        ----------
        k0 : float
            wave-number in air at a given frequency (omega/c0)
        inf_value : np1dArray or list
            values inferred of all parameters (Re and Im)
        theta : float or array
            angle of incidence in radians
        Returns
        ----------
        vp : complex
            complex-valued reflection coefficient
        alpha : floar
            absorption coefficient
        """
        # Get the samples of kp and rhop
        beta_samples = self.get_beta(ba.prior_samples[:,0:2])       
        # Angular wave-number in air
        k0z = k0*np.cos(theta)
        
        # Reflection coefficient samples
        vp_samples = (k0z - k0*beta_samples)/(k0z + k0*beta_samples)
        # absorption coefficient (samples)
        alpha_samples = 1-np.abs(vp_samples)**2
        # Use Bayesian object to compute statistics - Vp
        mu, ci = self.ba_recon(np.array([np.real(vp_samples), np.imag(vp_samples)]).T, 
                               ba.weights)    
        vp_mean = mu[0] + 1j*mu[1]
        vp_ci = np.zeros(2, dtype = complex)
        vp_ci[0] = ci[0, 0] + 1j*ci[0, 1]
        vp_ci[1] = ci[1, 0] + 1j*ci[1, 1]
        
        # Use Bayesian object to compute statistics - alpha
        mu, ci = self.ba_recon(np.array([alpha_samples]).T, ba.weights)
        alpha_mean = mu[0]
        alpha_ci = np.zeros(2)
        alpha_ci[0] = ci[0, 0]
        alpha_ci[1] = ci[1, 0]
        
        return vp_mean, vp_ci.T, alpha_mean, alpha_ci.T
    
    def get_vp_lr_angle(self, jf, ba, theta):
        # Initialization
        vp_mean = np.zeros(len(theta), dtype = complex)
        vp_ci = np.zeros((2, len(theta)), dtype = complex)
        alpha_mean = np.zeros(len(theta))        
        alpha_ci = np.zeros((2, len(theta)))
        # zs_mean = np.zeros(len(theta), dtype = complex)
        # zs_ci = np.zeros((2, len(theta)), dtype = complex)
        
        # Freq loop
        for jt, the in enumerate(theta):
            var = self.recon_vp_lr(self.controls.k0[jf], the, ba)
            vp_mean[jt] = var[0]
            vp_ci[:, jt] = var[1]
            alpha_mean[jt] = var[2]
            alpha_ci[:, jt] = var[3]
            # zs_mean[jt] = var[4]
            # zs_ci[:, jt] = var[5]
        return vp_mean, vp_ci.T, alpha_mean, alpha_ci#, zs_mean, zs_ci
    
    def get_vp_lr_spk(self, theta):
        # Initialization
        self.vp_mean = np.zeros((len(theta), len(self.controls.k0)), dtype = complex)
        self.vp_ci = np.zeros((2, len(theta), len(self.controls.k0)), dtype = complex)
        self.alpha_mean = np.zeros((len(theta), len(self.controls.k0)))        
        self.alpha_ci = np.zeros((2, len(theta), len(self.controls.k0)))
        self.zs_mean = np.zeros((len(theta), len(self.controls.k0)), dtype = complex)
        self.zs_ci = np.zeros((2, len(theta), len(self.controls.k0)), dtype = complex)
        
        # Freq loop
        bar = tqdm(total=len(self.controls.k0)*len(theta),
                   desc=r'Reconstruct $V_p$ (LR) spk.', ascii=False)
        for jt, the in enumerate(theta):
            for jf, k0 in enumerate(self.controls.k0):
                var = self.recon_vp_lr(k0, the, self.ba_list[jf])
                self.vp_mean[jt, jf] = var[0]
                self.vp_ci[:, jt, jf] = var[1]
                self.alpha_mean[jt, jf] = var[2]
                self.alpha_ci[:, jt, jf] = var[3]
                beta_mean = self.ba_list[jf].mean_values[0] +\
                    + 1j*self.ba_list[jf].mean_values[1]
                self.zs_mean[jt, jf] = (self.air.rho0*self.air.c0)/beta_mean
                beta_ci = self.ba_list[jf].ci[0,0] +\
                    + 1j*self.ba_list[jf].ci[0,1]
                self.zs_ci[0, jt, jf] = (self.air.rho0*self.air.c0)/beta_ci
                beta_ci = self.ba_list[jf].ci[1,0] +\
                    + 1j*self.ba_list[jf].ci[1,1]
                self.zs_ci[1, jt, jf] = (self.air.rho0*self.air.c0)/beta_ci
                bar.update(1)
        bar.close()
    
    def kp_rhop_range(self, resist = [3000, 60000], phi = [0.15, 1.0],
                      alpha_inf = [1.0, 3.00], Lam = [50e-6, 300e-6],
                      Lamlfac = [1.01, 3.0],
                      thickness = [2e-3, 20e-2], theta_deg = [0, 75],
                      n_samples = 20000):
        """ Run a JCA range study to determine useful ranges for kp and rhop
        """
        k_p_samples, rho_p_samples, beta_samples = mat.kp_rhop_range_study(
            resist = resist, phi = phi, alpha_inf = alpha_inf, Lam = Lam,
            Lamlfac = Lamlfac, thickness = thickness, theta_deg = theta_deg, 
            n_samples = n_samples, freq_vec = self.controls.freq)
        if self.chosen_model == 10:
            self.lb_mtx = mat.get_min_beta(beta_samples)
            self.ub_mtx = mat.get_max_beta(beta_samples)
        else:
            self.lb_mtx = mat.get_min_kp_rhop(k_p_samples, rho_p_samples)
            self.ub_mtx = mat.get_max_kp_rhop(k_p_samples, rho_p_samples)
            # self.lb_mtx[2,:] = 1.0#self.air.rho0*np.ones(self.lb_mtx.shape[1])
            # self.ub_mtx[2,:] = 1.1#*self.air.rho0*np.ones(self.lb_mtx.shape[1])
    
    def kp_rhop_range_del(self, resist = [3000, 60000],
                           thickness = [2e-3, 20e-2], theta_deg = [0, 75],
                           n_samples = 20000):
        """ Run a Delany-Bazley range study to determine useful ranges for kp and rhop
        """
        k_p_samples, rho_p_samples, beta_samples = mat.kp_rhop_range_study_del(
            resist = resist, thickness = thickness, theta_deg = theta_deg, 
            n_samples = n_samples, freq_vec = self.controls.freq)
        if self.chosen_model == 10:
            self.lb_mtx = mat.get_min_beta(beta_samples)
            self.ub_mtx = mat.get_max_beta(beta_samples)
        else:
            self.lb_mtx = mat.get_min_kp_rhop(k_p_samples, rho_p_samples)
            self.ub_mtx = mat.get_max_kp_rhop(k_p_samples, rho_p_samples)
            
    def kp_rhop_range_miki(self, resist = [3000, 60000],
                           thickness = [2e-3, 20e-2], theta_deg = [0, 75],
                           n_samples = 20000):
        """ Run a Miki range study to determine useful ranges for kp and rhop
        """
        k_p_samples, rho_p_samples, beta_samples = mat.kp_rhop_range_study_miki(
            resist = resist, thickness = thickness, theta_deg = theta_deg, 
            n_samples = n_samples, freq_vec = self.controls.freq)
        if self.chosen_model == 10:
            self.lb_mtx = mat.get_min_beta(beta_samples)
            self.ub_mtx = mat.get_max_beta(beta_samples)
        else:
            self.lb_mtx = mat.get_min_kp_rhop(k_p_samples, rho_p_samples)
            self.ub_mtx = mat.get_max_kp_rhop(k_p_samples, rho_p_samples)
    
    def plot_reim(self, meanval, cival, label = None, 
                  ax = None, figshape = (1, 2), figsize = (8, 3),
                  color = 'r', alpha = 1):
        """ Plots real and imaginary values of a quantity with its confidence interval
        """
        if ax is None:
            fig, ax = ut_is.give_me_an_ax(figshape, figsize)
            
        ax = ut_is.plot_spk_re_imag(self.controls.freq, meanval, ax = ax, 
                                    xlims = (self.controls.freq[0], self.controls.freq[-1]), 
                                    ylims = None, color = color, alpha = alpha, 
                                    linewidth = 1.5, linestyle = '-',
                                    label = label)
        ax[0,0].fill_between(self.controls.freq, np.real(cival[0,:]), 
                             np.real(cival[1,:]), color=color, alpha = 0.2,
                             edgecolor = 'none')
        ax[ax.shape[0]-1,ax.shape[1]-1].fill_between(self.controls.freq, 
                                                     np.imag(cival[0,:]), 
                                                     np.imag(cival[1,:]), 
                                                     color=color, alpha = 0.2,
                                                     edgecolor = 'none')
        ax[0,0].set_ylim((np.amin(np.real(cival[0,:])), 
                          np.amax(np.real(cival[1,:])))) 
        ax[ax.shape[0]-1,ax.shape[1]-1].set_ylim((np.amin(np.imag(cival[0,:])), 
                                                  np.amax(np.imag(cival[1,:]))))
        return ax
    
    def plot_kp(self, ax = None, figshape = (1, 2), figsize = (8, 3),
                color = 'r', alpha = 1.0):
        """ Plots the wave-number as a function of frequency
        """
        kp_mean = self.kp_mean/self.controls.k0
        ax = self.plot_reim(self.kp_mean, self.kp_ci, label = None, ax = ax,
                            figshape = figshape, figsize = figsize, color = color,
                            alpha = alpha)
        ax[0,0].set_ylabel(r"$Re\{k_p\}$")
        ax[ax.shape[0]-1,ax.shape[1]-1].set_ylabel(r"$Im\{k_p\}$")
        plt.tight_layout()
        return ax
    
    def plot_rhop(self, ax = None, figshape = (1, 2), figsize = (8, 3),
                  color = 'r', alpha = 1.0):
        """ Plots the density as a function of frequency
        """
        ax = self.plot_reim(self.rhop_mean, self.rhop_ci, label = None, ax = ax,
                            figshape = figshape, figsize = figsize, color = color,
                            alpha = alpha)
        ax[0,0].set_ylabel(r"$Re\{\rho_p\}$")
        ax[ax.shape[0]-1,ax.shape[1]-1].set_ylabel(r"$Im\{\rho_p\}$")
        plt.tight_layout()
        return ax
        
    def plot_Zp(self, ax = None, figshape = (1, 2), figsize = (8, 3),
                color = 'r', alpha = 1.0):
        """ Plots the characteristic impedance as a function of frequency
        """
        ax = self.plot_reim(self.zp_mean, self.zp_ci, label = None, ax = ax,
                            figshape = figshape, figsize = figsize, color = color,
                            alpha = alpha)
        ax[0,0].set_ylabel(r"$Re\{Z_p\}$")
        ax[ax.shape[0]-1,ax.shape[1]-1].set_ylabel(r"$Im\{Z_p\}$")
        plt.tight_layout()
        return ax
    
    def plot_Vp_spk(self, ax = None, figshape = (1, 2), figsize = (8, 3),
                color = 'r', alpha = 1.0, jtheta = 0):
        """ Plots the Reflection coefficient as a function of frequency
        """
        ax = self.plot_reim(self.vp_mean[jtheta, :], self.vp_ci[:, jtheta,:], 
                            label = None, ax = ax,
                            figshape = figshape, figsize = figsize, 
                            color = color, alpha = alpha)
        ax[0,0].set_ylim((-1,1))
        ax[ax.shape[0]-1,ax.shape[1]-1].set_ylim((-1,1))
        ax[0,0].set_ylabel(r"$Re\{V_p\}$")
        ax[ax.shape[0]-1,ax.shape[1]-1].set_ylabel(r"$Im\{V_p\}$")
        plt.tight_layout()
        return ax
    
    def plot_Zs_spk(self, ax = None, figshape = (1, 2), figsize = (8, 3),
                color = 'r', alpha = 1, jtheta = 0, normalize = True):
        """ Plots the Surface impedance as a function of frequency
        """
        if normalize:
            zs_mean = self.zs_mean[jtheta, :]/(self.air.rho0*self.air.c0)
            zs_ci = self.zs_ci[:,jtheta, :]/(self.air.rho0*self.air.c0)
            ylabel = [r"$Re\{Z_s\}/(\rho_0 c_0)$", r"$Im\{Z_s\}/(\rho_0 c_0)$"]
        else:
            zs_mean = self.zs_mean[jtheta, :]
            zs_ci = self.zs_ci[:,jtheta, :]
            ylabel = [r"$Re\{Z_s\}$", r"$Im\{Z_s\}$"]
            
        ax = self.plot_reim(zs_mean, zs_ci, 
                            label = None, ax = ax,
                            figshape = figshape, figsize = figsize, color = color,
                            alpha = alpha)
        ax[0,0].set_ylim((-1,1))
        ax[ax.shape[0]-1,ax.shape[1]-1].set_ylim((-1,1))
        ax[0,0].set_ylabel(ylabel[0])
        ax[ax.shape[0]-1,ax.shape[1]-1].set_ylabel(ylabel[1])
        plt.tight_layout()
        return ax
    
    def plot_alpha_spk(self, ax = None, figshape = (1, 2), figsize = (8, 3),
                color = 'r', alpha = 1, jtheta = 0):
        """ Plots the Absorption coefficient as a function of frequency
        """        
        ax = ut_is.plot_absorption(self.controls.freq, self.alpha_mean[jtheta, :], ax = ax, 
                                   xlim = (self.controls.freq[0], self.controls.freq[-1]), 
                                   ylim = (-0.2, 1.2), 
                                   color = color, linewidth = 1.5, linestyle = '-',
                                   alpha = alpha, label = None)
        ax.fill_between(self.controls.freq, self.alpha_ci[0, jtheta, :], 
                        self.alpha_ci[1, jtheta, :], color=color, alpha = 0.2,
                        edgecolor = 'none')
        ax.set_ylabel(r"$\alpha$ [-]")
        plt.tight_layout()
        return ax
    
    def plot_logZ_spk(self, ax = None, figshape = (1, 2), figsize = (8, 3),
                color = 'r', tolerance = 5, linestyle = '-', alpha = 1, label = ''):
        """ Plots the LogZ as a function of frequency
        """        
        ax = ut_is.plot_1d_curve(self.controls.freq, self.logZ_spk, ax = ax, 
                                 xlims = (self.controls.freq[0], self.controls.freq[-1]), 
                                 ylims = None,
                                 color = color, linewidth = 1.5, linestyle = linestyle,
                                 alpha = alpha, label = label,
                                 xlabel = "Frequency", ylabel = r"$\log(Z)$ [Np]",
                                 linx = False, liny = True, 
                                 xticks = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        
        ax.fill_between(self.controls.freq, self.logZ_spk - tolerance, 
                        self.logZ_spk + tolerance, color=color, alpha = 0.2,
                        edgecolor = 'none')
        # ax.set_ylabel(r"$\alpha$ [-]")
        plt.tight_layout()
        return ax
    
    def plot_search_space(self):
        """ plots the search space as a function of frequency (sanity check)
        """
        if self.chosen_model == 10 or self.chosen_model == 11:
            figshape = (1, 2)
            figsize = (8, 4)
            fig, ax = ut_is.give_me_an_ax(figshape, figsize)
            beta_l = self.lb_mtx[0,:] + 1j*self.lb_mtx[1,:]
            ax = ut_is.plot_spk_re_imag(self.controls.freq, beta_l, ax = ax, 
                                   xlims = (self.controls.freq[0], self.controls.freq[-1]), 
                                            ylims = None, 
                                            color = 'k', linewidth = 1.5, linestyle = '--',
                                            alpha = 1.0, label = None)
            beta_u = self.ub_mtx[0,:] + 1j*self.ub_mtx[1,:]
            ax = ut_is.plot_spk_re_imag(self.controls.freq, beta_u, ax = ax, 
                                        xlims = (self.controls.freq[0], self.controls.freq[-1]), 
                                        ylims = None, 
                                        color = 'k', linewidth = 1.5, linestyle = '--',
                                        alpha = 1.0, label = None)
            # Physical limits
            ax[0,0].plot(self.controls.freq, 0.001+np.zeros(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            # Fill
            ax[0,0].fill_between(self.controls.freq, np.real(beta_l), np.real(beta_u), 
                                 color='grey', alpha = 0.5)
            ax[0,1].fill_between(self.controls.freq, np.imag(beta_l), np.imag(beta_u), 
                                 color='grey', alpha = 0.5)
            # Labels
            ax[0,0].set_ylim((-0.1, np.amax(np.real(beta_u)))) 
            ax[0,1].set_ylim((np.amin(np.imag(beta_l)), np.amax(np.imag(beta_u))))
            ax[0,0].set_xlabel("Frequency [Hz]")
            ax[0,1].set_xlabel("Frequency [Hz]")
            ax[0,0].set_ylabel(self.parameters_names[0])
            ax[0,1].set_ylabel(self.parameters_names[1])
            plt.tight_layout()
        else:
            figshape = (2, 2)
            figsize = (8, 6)
            fig, ax = ut_is.give_me_an_ax(figshape, figsize)
            # plot kp
            kp_l = self.lb_mtx[0,:] + 1j*self.lb_mtx[1,:]
            ut_is.plot_1d_curve(self.controls.freq, np.real(kp_l), ax[0,0], 
                                     color = 'k', linestyle = '--', linx = False)
            ut_is.plot_1d_curve(self.controls.freq, np.imag(kp_l), ax[0,1], 
                                     color = 'k', linestyle = '--', linx = False)
            kp_u = self.ub_mtx[0,:] + 1j*self.ub_mtx[1,:]
            ut_is.plot_1d_curve(self.controls.freq, np.real(kp_u), ax[0,0], 
                                     color = 'k', linestyle = '--', linx = False)
            ut_is.plot_1d_curve(self.controls.freq, np.imag(kp_u), ax[0,1], 
                                     color = 'k', linestyle = '--', linx = False)
            
            # plot rhop
            rhop_l = self.lb_mtx[2,:] + 1j*self.lb_mtx[3,:]
            ut_is.plot_1d_curve(self.controls.freq, np.real(rhop_l), ax[1,0], 
                                     color = 'k', linestyle = '--', linx = False)
            ut_is.plot_1d_curve(self.controls.freq, np.imag(rhop_l), ax[1,1], 
                                     color = 'k', linestyle = '--', linx = False)
            rhop_u = self.ub_mtx[2,:] + 1j*self.ub_mtx[3,:]
            ut_is.plot_1d_curve(self.controls.freq, np.real(rhop_u), ax[1,0], 
                                     color = 'k', linestyle = '--', linx = False)
            ut_is.plot_1d_curve(self.controls.freq, np.imag(rhop_u), ax[1,1], 
                                     color = 'k', linestyle = '--', linx = False)
            
            # Physical limits
            ax[0,0].plot(self.controls.freq, np.ones(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            ax[0,1].plot(self.controls.freq, -0.01+np.zeros(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            ax[1,0].plot(self.controls.freq, 0.9*np.ones(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            ax[1,1].plot(self.controls.freq, -0.01+np.zeros(len(self.controls.freq)),
                         ':r', alpha = 0.6)
            # Fill
            ax[0,0].fill_between(self.controls.freq, np.real(kp_l), np.real(kp_u), 
                                 color='grey', alpha = 0.5)
            ax[0,1].fill_between(self.controls.freq, np.imag(kp_l), np.imag(kp_u), 
                                 color='grey', alpha = 0.5)
            ax[1,0].fill_between(self.controls.freq, np.real(rhop_l), np.real(rhop_u), 
                                 color='grey', alpha = 0.5)
            ax[1,1].fill_between(self.controls.freq, np.imag(rhop_l), np.imag(rhop_u), 
                                 color='grey', alpha = 0.5)
            # labels
            ax[0,0].set_ylim((0.5, np.amax(np.real(kp_u)))) 
            ax[0,1].set_ylim((np.amin(np.imag(kp_l)), 0.1))
            ax[1,0].set_ylim((0.9, np.amax(np.real(rhop_u)))) 
            ax[1,1].set_ylim((np.amin(np.imag(rhop_l)), 0.1))
            ax[1,0].set_xlabel("Frequency [Hz]")
            ax[1,1].set_xlabel("Frequency [Hz]")
            ax[0,0].set_ylabel(self.parameters_names[0] + r'$/k_0$')
            ax[0,1].set_ylabel(self.parameters_names[1] + r'$/k_0$')
            ax[1,0].set_ylabel(self.parameters_names[2] + r'$/\rho_0$')
            ax[1,1].set_ylabel(self.parameters_names[3] + r'$/\rho_0$')
            plt.tight_layout()
            
    def save(self, filename = 'dcism_ba', path = ''):
        """ To save the decomposition object as pickle
        """
        ut_is.save(self, filename = filename, path = path)

    def load(self, filename = 'dcism_ba', path = ''):
        """ To load the decomposition object as pickle

        You can instantiate an empty object of the class and load a saved one.
        It will overwrite the empty object.
        """
        ut_is.load(self, filename = filename, path = path)
            
    
    
    
    
    
    # available_models = {0:
    #                     {"name": "Locally reacting sample - H(f)",
    #                      "num_model_par": 2,
    #                      "parameters_names" : [r"$Re\{\beta\}$", r"$Im\{\beta\}$"],
    #                      'known_thickness': True,
    #                      'TF' : True},
    #                     1:
    #                     {"name": "Locally reacting sample - P(f)",
    #                      "num_model_par": 4,
    #                      "parameters_names" : [r"$Re\{\beta\}$", r"$Im\{\beta\}$", 
    #                                            r"$|S|$", r"$\angle S$"],
    #                      'known_thickness': True,
    #                      'TF' : False},                   
    #                     2:
    #                     {"name": "NLR semi-infinite layer - H(f)",
    #                      "num_model_par": 4,
    #                      "parameters_names" : [r"$Re\{k_p/k_0\}$", r"$Im\{k_p/k_0\}$",
    #                                            r"$Re\{\rho_p/\rho_0\}$", 
    #                                            r"$Im\{\rho_p/\rho_0\}$"],
    #                      'known_thickness': True,
    #                      'TF' : True},
    #                     3:
    #                     {"name": "NLR semi-infinite layer - P(f)",
    #                      "num_model_par": 6,
    #                      "parameters_names" : [r"$Re\{k_p/k_0\}$", r"$Im\{k_p/k_0\}$",
    #                                            r"$Re\{\rho_p/\rho_0\}$", 
    #                                            r"$Im\{\rho_p/\rho_0\}$",
    #                                            r"$|S|$", r"$\angle S$"],
    #                      'known_thickness': True,
    #                      'TF' : False},
    #                     4:
    #                     {"name": "NLR single layer with known thickness - H(f)",
    #                      "num_model_par": 4,
    #                      "parameters_names" : [r"$Re\{k_p/k_0\}$", r"$Im\{k_p/k_0\}$",
    #                                            r"$Re\{\rho_p/\rho_0\}$", 
    #                                            r"$Im\{\rho_p/\rho_0\}$"],
    #                      'known_thickness': True,
    #                      'TF' : True},
    #                     5:
    #                     {"name": "NLR single layer with unknown thickness - H(f)",
    #                      "num_model_par": 5,
    #                      "parameters_names" : [r"$Re\{k_p/k_0\}$", r"$Im\{k_p/k_0\}$",
    #                                            r"$Re\{\rho_p/\rho_0\}$", 
    #                                            r"$Im\{\rho_p/\rho_0\}$", 
    #                                            r"$t_p$"],
    #                      'known_thickness': False,
    #                      'TF' : True},
    #                     6:
    #                     {"name": "NLR single layer with known thickness - P(f)",
    #                      "num_model_par": 6,
    #                      "parameters_names" : [r"$Re\{k_p/k_0\}$", r"$Im\{k_p/k_0\}$",
    #                                            r"$Re\{\rho_p/\rho_0\}$", r"$Im\{\rho_p/\rho_0\}$",
    #                                            r"$|S|$", r"$\angle S$"],
    #                      'known_thickness': True,
    #                      'TF' : False},
    #                     7:
    #                     {"name": "NLR single layer with unknown thickness - P(f)",
    #                      "num_model_par": 7,
    #                      "parameters_names" : [r"$Re\{k_p/k_0\}$", r"$Im\{k_p/k_0\}$",
    #                                            r"$Re\{\rho_p/\rho_0\}$", r"$Im\{\rho_p/\rho_0\}$",
    #                                            r"$|S|$", r"$\angle S$",  
    #                                            r"$t_p$"],
    #                      'known_thickness': False,
    #                      'TF' : False},
    #                     8:
    #                     {"name": "PW-NLR single layer with known thickness - H(f)",
    #                      "num_model_par": 4,
    #                      "parameters_names" : [r"$Re\{k_p/k_0\}$", r"$Im\{k_p/k_0\}$",
    #                                            r"$Re\{\rho_p/\rho_0\}$", 
    #                                            r"$Im\{\rho_p/\rho_0\}$"],
    #                      'known_thickness': True,
    #                      'TF' : True}
    #                     }