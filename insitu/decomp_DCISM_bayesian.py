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
                 receivers = None, source = None, sampling_scheme = 'slice'):
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
        # self.set_reference_sensor(ref_sens = 0)
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
                            {"name": "NLR semi-infinite layer",
                             "num_model_par": 6,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$", 
                                                   r"$|Q|$", r"$\angle Q$"]},
                            2:
                            {"name": "NLR layer with known thickness",
                             "num_model_par": 6,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$", 
                                                   r"$|Q|$", r"$\angle Q$"]},
                            3:
                            {"name": "NLR layer with known thickness (no Q)",
                             "num_model_par": 4,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$"]},
                            4:
                            {"name": "NLR layer with uncertain thickness",
                             "num_model_par": 7,
                             "parameters_names" : [r"$Re\{k_p\}$", r"$Im\{k_p\}$",
                                                   r"$Re\{\rho_p\}$", r"$Im\{\rho_p\}$",
                                                   r"$t_p$", r"$|Q|$", r"$\angle Q$"]},
                            
                            
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
            
    def forward_model_2(self, x_meas, model_par = [3.00, -56.00, 1.20, -64.00, 1e-5, -np.pi]):
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
        # vol_velocity = model_par[4]*np.exp(1j*model_par[5])
        # source_strength = 1j*self.air.rho0*self.controls.c0*self.current_k0*vol_velocity
        source_strength = model_par[4]*np.exp(1j*model_par[5])
        # complex wave-number
        k_p = model_par[0] + 1j*model_par[1]
        # complex density
        rho_p = model_par[2] + 1j*model_par[3]
        green_fun = self.dDCISMsf.predict_p(k = self.current_k0, k_p = k_p, rho_p = rho_p, 
                                            t_p = self.t_p)
        p_pred = source_strength * green_fun
        return p_pred
    
    def forward_model_3(self, x_meas, model_par = [3.00, -56.00, 1.20, -64.00]):
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
        # source_strength = 1.0
        # complex wave-number
        k_p = self.current_k0*(model_par[0] + 1j*model_par[1])
        # complex density
        rho_p = self.air.rho0*(model_par[2] + 1j*model_par[3])
        green_fun = self.dDCISMsf.predict_p(k = self.current_k0, k_p = k_p, rho_p = rho_p, 
                                            t_p = self.t_p)
        p_pred = green_fun / green_fun[self.ref_sens]
        p_pred = np.delete(p_pred, self.ref_sens)
        # p_pred = source_strength * green_fun
        return p_pred
    
    def forward_model_33(self, x_meas, model_par = [3.00, -56.00, 1.20, -64.00]):
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
        # source_strength = 1.0
        # complex wave-number
        k_p = self.current_k0*(model_par[0] + 1j*model_par[1])
        # complex density
        rho_p = self.air.rho0*(model_par[2] + 1j*model_par[3])
        green_fun = self.dDCISMsf.predict_p(k = self.current_k0, k_p = k_p, rho_p = rho_p, 
                                            t_p = self.t_p)
        
        p_exp_line_1 = green_fun[self.id_z_list[0]]
        p_exp_line_2 = green_fun[self.id_z_list[1]]
        p_pred = p_exp_line_1/p_exp_line_2
        
        # p_pred = green_fun / green_fun[self.ref_sens]
        # p_pred = np.delete(p_pred, self.ref_sens)
        # p_pred = source_strength * green_fun
        return p_pred
    
    def nested_sampling_single_freq(self, jf = 0, n_live = 250, max_iter = 10000, 
                    max_up_attempts = 100, seed = 0):
        """ Run single freq inference
        
        Parameters
        ----------
        jf : int
            Freq index to run
        n_live : int
            The number of live points in your initial and final population.
            It will contain your initial population, and be updated. As the iterations
            go on, the likelihood of such set of points will increase until termination.
        max_iter : int
            The maximum number of iterations allowed.
        tol_evidence_increase : float
            The tolerance of evidence increase - used for iteration stop. 
            If the maximum log-likelihood multiplied by the current mass 
            does not help to increase the evidence*tol_evidence_increase,
            then the iterations can stop.
        seed : int
            seed for random number generator.
        max_up_attempts : int
            number of attempts to update the parameter set of lowest log-likelihood
            to a parameter set with higher likelihood.
        """
        self.current_k0 = self.controls.k0[jf]
        ba = BayesianSampler(measured_coords = self.receivers.coord[self.id_z_list[0],:], 
                              measured_data = self.pres_s[:, jf],
                              parameters_names = self.parameters_names,
                              num_model_par = self.num_model_par,
                              sampling_scheme = self.sampling_scheme)
        ba.set_model_fun(model_fun = self.forward_model_33)
        ba.set_uniform_prior_limits(lower_bounds = self.lower_bounds, 
                                    upper_bounds = self.upper_bounds)
        # ba.nested_sampling(n_live = n_live, max_iter = max_iter, 
        #                    max_up_attempts = max_iter, seed = seed)
        ba.ultranested_sampling(n_live = n_live, max_iter = max_iter)
        ba.compute_statistics()
        return ba
    
    def pk_bayesian(self, n_live = 250, max_iter = 10000, 
                    max_up_attempts = 100, seed = 0):
        
        self.ba_list = []
        
        # bar = tqdm(total=len(self.controls.k0),
        #            desc='Calculating Bayesian inversion (dDCISM)...', ascii=False)
        
        # Freq loop
        for jf, self.current_k0 in enumerate(self.controls.k0):
            print("Baysian inversion for freq {} [Hz]. {} of {}".format(self.controls.freq[jf],
                                                                        jf+1, len(self.controls.k0)))
            ba = self.nested_sampling_single_freq(jf = jf, n_live = n_live, 
                                                  max_iter = max_iter,
                                                  max_up_attempts = max_up_attempts, 
                                                  seed = seed)
            self.ba_list.append(ba)
        #     bar.update(1)
        # bar.close()
        
        
    def get_kp_rhop(self, ba, mode = "mean"):
        if mode == "mean":
            kp = self.current_k0*(ba.mean_values[0] + 1j*ba.mean_values[1])
            rhop = self.air.rho0*(ba.mean_values[2] + 1j*ba.mean_values[3])
        elif mode == "lower_ci":
            kp = self.current_k0*(ba.ci[0,0] + 1j*ba.ci[0,1])
            rhop = self.air.rho0*(ba.ci[0,2] + 1j*ba.ci[0,3])
        elif mode == "upper_ci":
            kp = self.current_k0*(ba.ci[1,0] + 1j*ba.ci[1,1])
            rhop = self.air.rho0*(ba.ci[1,2] + 1j*ba.ci[1,3])
        else:
            kp = self.current_k0*(ba.mean_values[0] + 1j*ba.mean_values[1])
            rhop = self.air.rho0*(ba.mean_values[2] + 1j*ba.mean_values[3])
        return kp, rhop
    
    def recon_mat_props_nlr(self, ba, id_f, theta, mode = "mean"):
        """ From infereed kp and rhop, reconstruct Zp, cp
        
        Single frequency estimation (can be multiple angle)
        
        Parameters
        ----------
        ba : object
            Bayesian inference object
        id_f : int
            freq index
        theta : float or numpy1dArray
            angle of incidence in radians
        """
        # Chose thickness according to model
        if self.chosen_model == 4:
            tp = ba.mean_values[4]
        else:
            tp = self.t_p
        # Wave-number and density from mean values
        kp, rhop = self.get_kp_rhop(ba, mode = mode)
        # Characteristic and surface impedance
        Zp, Zs = self.imp_recon_nlr(w = self.controls.w[id_f], kp = kp,
                                    rhop = rhop, theta = theta, tp = tp)
        # Reflection and absorption coefficient
        Vp, alpha = self.ref_abs_coeff(Zs, theta)
        return kp, rhop, Zp, Zs, Vp, alpha
    
    def imp_recon_nlr(self, w, kp, rhop, theta, tp):
        """ Reconstruct Zp and Zs
        
        Parameters
        ----------
        w : float
            angular frequency in rad/s
        kp : float complex
            Wave-number in porous media. It can come from mean or confidence interval
        rhop : float complex
            Density in porous media. It can come from mean or confidence interval
        """
        # Characteristic impedance
        Zp = rhop * w/kp
        # k0
        k0 = w/self.air.c0
        # refraction index
        n_index = kp/k0
        theta_t = np.arcsin(np.sin(theta)/n_index)
        kzp = kp * np.cos(theta_t)
        Zs = -1j * Zp * (kp/kzp) *(1/np.tan(kzp*tp))
        return Zp, Zs
    
    def ref_abs_coeff(self, Zs, theta):
        """ From infereed Zs, reconstruct Vp, alpha
        
        Single frequency estimation (can be multiple angle)
        
        Parameters
        ----------
        ba : object
            Bayesian inference object
        theta : float or numpy1dArray
            angle of incidence in radians
        """
        # Reflection and absorption coefficient
        Vp = (Zs*np.cos(theta) - (self.air.rho0 * self.air.c0)) /\
            (Zs*np.cos(theta) + (self.air.rho0 * self.air.c0))
        alpha = 1-(np.abs(Vp))**2
        return Vp, alpha