# -*- coding: utf-8 -*-
"""
Created on Sep  2025

@author: Eric Brandão - Bayesian sampler class
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils_insitu as ut_is

class BayesianSampler(object):
    """ Bayesian sampling analysis of inverse problems
    """
    
    def __init__(self, measured_coords = None, measured_data = None, 
                 num_model_par = 1, parameters_names = None, seed = 42):
        """ Bayesian sampling schemes
        
        Parameters
        ----------
        measured_data : numpy1dArray
            Array containing the measured data 
        model_fun : function callable
            Function containing your model. Will be called multiple times during inference
            Its input are the measured coordinates and the
        """
        # Make sure there is measured data
        if measured_coords is None or measured_data is None:
            raise ValueError("It is necessary to input your measured coordinates and data. Otherwise, I can not do anything!")
        elif measured_coords.shape[0] != len(measured_data):
            raise ValueError("Measured coordinates length and data length must be the same!")
        # Make sure parameters name has the same size of num_par
        elif len(parameters_names) != num_model_par:
            raise ValueError("The parameters_names list must be a list with same size of num_par")
        else:
            self.measured_coords = measured_coords
            self.measured_data = measured_data
            self.num_of_meas = len(measured_data)
            self.num_model_par = num_model_par
            self.parameters_names = parameters_names
            self.seed = seed
            # self.rng = np.random.default_rng(seed)
        # Make sure there is model function and get number of parameters to estimate
        # if not callable(model_fun):    
        #     raise ValueError("model_fun must be a callable function!")
        # else:
        #     self.model_fun = model_fun
        #     self.num_par = self.model_fun.__code__.co_argcount - 1
            
    def set_model_fun(self, model_fun):
        """ Set the model function which will be called multiple times at inference
        
        This function is called to evaluate the likelihood for a given set of model parameters.
        The function itself should have a structured input such as
        
        "def line_model(x_meas, model_par = [1, 0])"
        
        where x_meas stands for the measurement coordinates and model_par is a vector containing
        the guesses for the values in the parameter space. *kwargs may also apply depending
        on the model. The output should return a y_pred (prediction at x_coord)
        
        Parameters
        ----------
        model_fun : function callable
            Function containing your model. Will be called multiple times during inference
            Its input are the measured coordinates and the
        """
        if not callable(model_fun):    
            raise ValueError("model_fun must be a callable function!")
        else:
            self.model_fun = model_fun
        
            
    def set_uniform_prior_limits(self, lower_bounds = [0], upper_bounds = [1]):
        """ Set the uniform prior limits
        
        Parameters
        ----------
        lower_bonds : numpy1dArray
            Array containing the lower bonds of the parameter space
        upper_bonds : numpy1dArray
            Array containing the upper bonds of the parameter space
        """
        # Make sure lower / upper bonds has the same size of num_par
        if len(lower_bounds) != self.num_model_par or len(upper_bounds) != self.num_model_par:
            raise ValueError("Lower and Upper bonds must be a vector with the same size of num_par")
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
    def set_convergence_tolerance(self, convergence_tol = [0.01]):
        """ Set convergence tolerance for nested sampling
        
        Parameters
        ----------
        convergence_tol : numpy1dArray
            Array containing the tolerance values for each of the model's parameters
        """
        # Make sure lower / upper bonds has the same size of num_par
        if len(convergence_tol) != self.num_model_par:
            raise ValueError("convergence_tol must be a vector with the same size of num_par")
        self.convergence_tol = convergence_tol
    
    def sample_uniform_prior(self, num_samples = 1):
        """ Sample uniform prior - all that is available for now
        
        Parameters
        ----------
        num_samples : int
            number of sampling points
        
        Returns
        ----------
        prior_samples : numpyndArray
            samples of the uniform prior distribution with dimensions (num_pts, num_model_par)
        """
        prior_samples = np.random.uniform(low = self.lower_bounds, 
                                          high = self.upper_bounds,
                                          size = (num_samples, self.num_model_par))
        return prior_samples
    
    def log_normal(self, model_par):
        """ Computes log-like Normal likelihood
        
        Parameters
        ----------
        model_par : numpy1dArray
            parameters for a giving tried solution vector
            
        Returns
        ----------
        logp : float
            Value of log - like Student-t distribution
        
        """
        # mu = np.array([1.0, -1.0])
        # d = model_par - mu
    
        # Sigma = np.array([[1.0, 0.8],
        #           [0.8, 1.5]])
        # invSigma = np.linalg.inv(Sigma)
        # return -0.5 * d @ invSigma @ d
        sigma = 1
        y_pred = self.model_fun(x_meas = self.measured_coords, 
                                model_par = model_par)
        # error norm
        error_norm = np.linalg.norm(self.measured_data - y_pred)
        logp = np.log(1/(sigma*np.sqrt(2*np.pi))) - error_norm**2/(2*sigma**2)
        return logp
        
    def log_t(self, model_par):
        """ Computes log-like Student-t likelihood
        
        Parameters
        ----------
        model_par : numpy1dArray
            parameters for a giving tried solution vector
            
        Returns
        ----------
        logp : float
            Value of log - like Student-t distribution
        
        """
        # prediction
        y_pred = self.model_fun(x_meas = self.measured_coords, 
                                model_par = model_par)
        # plt.plot(self.measured_coords, 10*np.log10(y_pred/np.amax(y_pred)), 'o')
        # error norm
        error_norm = np.linalg.norm(self.measured_data - y_pred)
        logp = -(self.num_of_meas/2)*np.log((error_norm**2)/2)
        return logp
    
    def brute_force_sampling_bar(self, num_samples = 1000):
        """ Brute force sample the parameter space.
        
        Parameters
        ----------
        num_samples : int
            total number of samples to draw
        """
        self.prior_samples = self.sample_uniform_prior(num_samples = num_samples)
        # Initialize logp
        logp = np.zeros(num_samples)
        # bar
        bar = tqdm(total = num_samples,
                   desc='Brute force sampling...', ascii=False)
        for jp, pr_sam in enumerate(self.prior_samples):
            logp[jp] = self.log_t(model_par = pr_sam)
            bar.update(1)
        bar.close()
        
        # Compute posterior weights ~ L * prior (prior is uniform, so just L)
        self.weights = np.exp(logp - np.max(logp))
        self.weights /= np.sum(self.weights)
        return self.prior_samples, self.weights, logp      
    
    def brute_force_sampling(self, num_samples = 1000):
        """ Brute force sample the parameter space.
        
        Parameters
        ----------
        num_samples : int
            total number of samples to draw
        """
        self.prior_samples = self.sample_uniform_prior(num_samples = num_samples)
        # Initialize logp
        logp = np.zeros(num_samples)
        for jp, pr_sam in enumerate(self.prior_samples):
            logp[jp] = self.log_t(model_par = pr_sam)
        
        # Compute posterior weights ~ L * prior (prior is uniform, so just L)
        self.weights = np.exp(logp - np.max(logp))
        self.weights /= np.sum(self.weights)
        return self.prior_samples, self.weights, logp 
    
    def update_values(self, random_par, adjust_suggested, current_worst_par_val,
                      type_of_update = "adding"):
        """ Update a given parameter value and logp from a set of parameter vector 
        """
        
        # Update (Adding)
        if type_of_update == "adding":
            updated_sample_par = current_worst_par_val[random_par] + adjust_suggested
        else:
            updated_sample_par = current_worst_par_val[random_par] - adjust_suggested
        updated_model_par = current_worst_par_val
        updated_model_par[random_par] = updated_sample_par
        updated_logp = self.log_t(updated_model_par)
        return updated_logp, updated_model_par       
    
    
    def get_worst_logp(self, it_num = 0):
        """ Find the worst logp value in live pts
        
        Finds the worst logp value in the current live pts population, then fill this
        logp value and associated parameters to the correct place in the dead points.
        
        This is Step 1 (Beaton and Xiang - Sec. IV.B.1)
        
        Parameters
        ----------
        it_num : int
            iteration number
        """
        # Find index of worst log-likelihood and update dead_pts and logp_dead
        self.worst_logp_index = np.argmin(self.logp_live)
        self.logp_dead[it_num] = self.logp_live[self.worst_logp_index]
        self.dead_pts[it_num, :] = self.live_pts[self.worst_logp_index, :]

    def compute_spread(self, pop_pts):
        """ Compute the spread (max-min) of a population of sampled values
        
        Parameters
        ----------
        pop_pts : numpyndArray
            A population of points in parameter space
        """
        spread = np.amax(pop_pts, axis = 0) - np.amin(pop_pts, axis = 0)
        return spread    
    
    def suggest_random_adjustment(self, spread):
        """ Suggest a random adjustment for a single random parameter
        
        These are Steps 2-3 (Beaton and Xiang - Sec. IV.B.1)
        
        1. Select a random parameter in the reference point.
        2. Use a uniform distribution to generate a random adjustment
        value for the parameter. The uniform distribution
        will have limits from zero to a maximum value dependent
        on the type of parameter selected. The MAX-MIN (spread/2) of the
        current live_pts population is used as suggestion for this update.
        
        Parameters
        ----------
        spread : numpy1dArray
            The Array containing the spread of the current population
        """
        # Select a random parameter index
        random_par_id = np.random.randint(low = 0, high = self.num_model_par, size = 1)[0]
        # Generate a random adjustment value for the parameter (uniform dist)
        # random_par_id = 0
        suggested_adjustment = np.random.uniform(low = 0,
                                                 high = spread[random_par_id]/2, 
                                                 size = 1)[0]
        # print("ID {} / Spread {:.3f} / Val. {:.3f}".format(random_par_id, spread[random_par_id], suggested_adjustment))
        # print(adjust_suggested)
        return random_par_id, suggested_adjustment
    
    def update_parameter(self, parameter_vec, 
                         random_par_id = 0, suggested_adjustment = 0.0):
        """ Updates the choosen parameter.
        
        We need to decide if we add or subtract the random suggestion. To do so, 
        we will calculate which is the closest border of the parameter we want to alter.
        If the parameter is closer to the lower bound we add. If the parameter is closer
        to the upper bound we subtract.
        
        This is Step 4 (Beaton and Xiang - Sec. IV.B.1)
        
        Parameters
        ----------
        parameter_vec : numpy1dArray
            Parameter vector to adjust
        random_par_id : int
            Parameter to update
        suggested_adjustment : float
            Value to add or subtract from current parameter.
        """
        suggested_parameter_vec = np.copy(parameter_vec)
        lower_bound_distance = parameter_vec[random_par_id] - self.lower_bounds[random_par_id]
        upper_bound_distance = self.upper_bounds[random_par_id] - parameter_vec[random_par_id]
        if lower_bound_distance <= upper_bound_distance:
            # print("Summing")
            suggested_parameter_vec[random_par_id] += np.abs(suggested_adjustment)
        else:
            # print("Subtracting")
            suggested_parameter_vec[random_par_id] -= np.abs(suggested_adjustment)
        return suggested_parameter_vec
    
    def constrained_resample2(self, current_worst_samplevec, current_spread,
                              current_worst_logp = -1e6, max_attempts = 5):
        """ Sample movement in the parameter space.
        
        Parameters
        ----------
        """
        # initialize while loop variables
        attempt_num = 0 # number of attempts made
        update_unsuccessful = True # success of update is set to False first
              
        # worst_logp = current_worst_logp-1
        while attempt_num <= max_attempts and update_unsuccessful:
            # Get a adjustment suggestion
            random_par_id, suggested_adjustment = self.suggest_random_adjustment(current_spread)
            self.random_par_id.append(random_par_id)
            # print("Att num {} / random id {} / random val {:.4f}".format(attempt_num,
            #                                                              random_par_id, suggested_adjustment))
            # Get a parameter suggestion
            suggested_parameter_vec =\
                self.update_parameter(parameter_vec = current_worst_samplevec,
                                      random_par_id = random_par_id, 
                                      suggested_adjustment = suggested_adjustment)
            # Evaluate logp of suggested parameter vector
            logp_new = self.log_t(model_par = suggested_parameter_vec)
            # print(attempt_num)
            if logp_new > current_worst_logp:
                # current_worst_logp =  np.copy(logp_new)
                # current_worst_samplevec = np.copy(suggested_parameter_vec)
                update_unsuccessful = False
            else:
                logp_new = np.copy(current_worst_logp)
                suggested_parameter_vec = np.copy(current_worst_samplevec)
            
            # if random_par_id == 1:
            #     print("Att num {} / random id {} / UnSuccess {:.4f}".format(attempt_num,
            #                                                             random_par_id, 
            #                                                             update_unsuccessful))
                
            # print("Att num {} / unSucess is {}".format(attempt_num, update_unsuccessful))
            # print("Att num {} / current worst logp {} / new logp {}".format(attempt_num,
            #                                                        current_worst_logp, logp_new))
            
            attempt_num += 1
            return logp_new, suggested_parameter_vec, update_unsuccessful
    
    def evaluate_move(self, logp_new, parameter_vec_new):
        """ Evaluate if the proposed move satisfies increase in likelihood
        
        Evaluate if the proposed move generated a higher likelihood. If so,
        then substitute the new logp to the worst one and the new parameter vector
        to the worst one.
        Parameters
        ----------
        
        """
        if logp_new > self.logp_live[self.worst_logp_index]:
            self.logp_live[self.worst_logp_index] = logp_new
            self.live_pts[self.worst_logp_index, :] = np.copy(parameter_vec_new)
            # update_unsuccessful = False
            self.update_successful = True
        
    
    def update_from_lowest_likelihood(self, current_spread, max_up_attempts):
        """ Update the parameter vector of lowest likelihood
        
        Updates the parameter of lowest likelihood and computes the new log likelihood.
        
        Parameters
        ----------
        current_spread : numpy1dArray
            The Array containing the spread of the current population
        
        Returns
        ----------
        logp_new : float
            New log-likelihood value
        parameter_vec_new : numpy 1dArray
            New samples value
        """
        attempt_num = 0 # number of attempts made
        while attempt_num <= max_up_attempts and not self.update_successful:
            # Get a random adjustment suggestion        
            random_par_id, suggested_adjustment = self.suggest_random_adjustment(current_spread)
            self.random_par_id.append(random_par_id)
            # Get a parameter vec suggestion
            parameter_vec_new =\
                self.update_parameter(parameter_vec = self.live_pts[self.worst_logp_index],
                                      random_par_id = random_par_id, 
                                      suggested_adjustment = suggested_adjustment)
            # Evaluate logp of suggested parameter vector
            logp_new = self.log_t(model_par = parameter_vec_new)
            # Evaluate if the move increased likelihood
            self.evaluate_move(logp_new, parameter_vec_new)
            attempt_num += 1
        # return logp_new, parameter_vec_new
        
    def update_rw(self, pt_of_origin, lower_bounds, upper_bounds,
                  max_up_attempts):
        """ Update the parameter vector of lowest likelihood (Random walk)
        
        Updates the parameter of lowest likelihood and computes the new log likelihood.
        
        Parameters
        ----------
        current_spread : numpy1dArray
            The Array containing the spread of the current population
        
        Returns
        ----------
        logp_new : float
            New log-likelihood value
        parameter_vec_new : numpy 1dArray
            New samples value
        """
        attempt_num = 0 # number of attempts made
        # Origin of random walk - live_pts with lowest likelihood
        parameter_vec_new = np.copy(pt_of_origin)
        while attempt_num <= max_up_attempts and not self.update_successful:
            # Get a random adjustment by random waling the original point
            parameter_vec_new, _ =\
                ut_is.random_move(origin = parameter_vec_new, 
                                  lower_bounds = lower_bounds, 
                                  upper_bounds = upper_bounds)
            # Evaluate logp of suggested parameter vector
            logp_new = self.log_t(model_par = parameter_vec_new)
            # Evaluate if the move increased likelihood
            self.evaluate_move(logp_new, parameter_vec_new)
            attempt_num += 1
        # return logp_new, parameter_vec_new
    
    def update_from_random_par_vec(self, current_spread):
        """ Update from a random parameter vector index
        """
        
        all_indexes = np.arange(0, self.live_pts.shape[0])
        remaining_indexes = np.delete(all_indexes, self.worst_logp_index)
        np.random.shuffle(remaining_indexes)
        # print(remaining_indexes)
        # update_successful = False
        i = 0
        while i < len(remaining_indexes) and not self.update_successful:
            # Get a adjustment suggestion
            random_par_id, suggested_adjustment = self.suggest_random_adjustment(current_spread)
            self.random_par_id.append(random_par_id)
            # Get a parameter suggestion
            # print(remaining_indexes[i])
            parameter_vec_new =\
                self.update_parameter(parameter_vec = self.live_pts[remaining_indexes[i]],
                                      random_par_id = random_par_id, 
                                      suggested_adjustment = suggested_adjustment)
            # Evaluate logp of suggested parameter vector
            logp_new = self.log_t(model_par = parameter_vec_new)
            self.evaluate_move(logp_new, parameter_vec_new)
            # print(i)
            i += 1
    
    def constrained_resample(self, i, max_up_attempts = 5):
        """ Sample movement in the parameter space.
        
        Parameters
        ----------
        """
        # Compute current spread in the given live_pts population
        current_spread = self.compute_spread(self.live_pts)
        # print(current_spread)
        self.spread[i,:] = current_spread
        # initialize while loop variables
        # attempt_num = 0 # number of attempts made
        # update_unsuccessful = True # success of update is set to False first
        self.update_successful = False # success of update is set to False first
        
        # worst_logp = current_worst_logp-1
        # while attempt_num <= max_up_attempts and not self.update_successful:
        # while self.update_successful:
        self.update_from_lowest_likelihood(current_spread, max_up_attempts)
        if not self.update_successful:
            self.update_from_random_par_vec(current_spread)
            
    def constrained_resample_rw(self, i, max_up_attempts = 5):
        """ Sample movement in the parameter space with random walk.
        
        Parameters
        ----------
        """
        # Compute current spread in the given live_pts population
        current_spread = self.compute_spread(self.live_pts)
        # print(current_spread)
        self.spread[i,:] = current_spread
        lower_bounds = 1.00*np.amin(self.live_pts, axis = 0)
        upper_bounds = 1.00*np.amax(self.live_pts, axis = 0)
        # initialize while loop variables
        self.update_successful = False # success of update is set to False first
        
        # worst_logp = current_worst_logp-1
        # while attempt_num <= max_up_attempts and not self.update_successful:
        # while self.update_successful:
        self.update_rw(pt_of_origin = self.live_pts[self.worst_logp_index], 
                       lower_bounds = lower_bounds, upper_bounds = upper_bounds,
                       max_up_attempts = max_up_attempts)
        if not self.update_successful:
            # Sample a random point in the current pop of live pts
            all_indexes = np.arange(0, self.live_pts.shape[0])
            remaining_indexes = np.delete(all_indexes, self.worst_logp_index)
            
            pt_of_origin_id = np.random.randint(low = 0, high = len(remaining_indexes),
                                                size = 1)[0]
            pt_of_origin = self.live_pts[remaining_indexes[pt_of_origin_id],:]
            # print(pt_of_origin)
            self.update_rw(pt_of_origin = pt_of_origin, 
                           lower_bounds = lower_bounds, upper_bounds = upper_bounds,
                           max_up_attempts = max_up_attempts)
            
    def slice_step(self, it_num, theta, win_size = 0.2, max_up_attempts = 200):
        """ Slice step function
        
        Parameters
        ----------
        it_num : int
            iteration number
        """
        theta = np.copy(theta)
        # direction = rng.normal(size = self.num_model_par)
        # direction /= np.linalg.norm(v)
        direction = ut_is.get_random_direction(num_dim = self.num_model_par)
        u_min, u_max = -0.5*win_size, 0.5*win_size
        for _ in range(max_up_attempts):
            u = np.random.uniform(low = u_min, high = u_max, size = 1)#rng.uniform(u_min, u_max)
            theta_trial = theta + u * direction
            if np.any(theta_trial < self.lower_bounds) or np.any(theta_trial > self.upper_bounds):
                if u > 0:
                    u_max = u
                else: 
                    u_min = u
                continue
            new_logp = self.log_t(model_par = theta_trial)
            if new_logp > self.logp_dead[it_num]:
                return theta_trial
            else:
                if u > 0: 
                    u_max = u
                else: 
                    u_min = u
        return theta
    
    def constrained_resample_slice(self, it_num = 0, win_size = 0.2, 
                                   max_up_attempts = 200):
        """ Constrained move using slice sampling
        """
        current_spread = self.compute_spread(self.live_pts)
        win_size = np.amax(current_spread)
        # print(current_spread)
        self.spread[it_num,:] = current_spread
        # Init new param
        theta_new = None
        for _ in range(200):
            cand = self.live_pts[np.random.randint(self.live_pts.shape[0]),:]
            theta_try = self.slice_step(it_num = it_num, theta = cand, win_size = win_size,
                                        max_up_attempts = max_up_attempts)
            new_logp = self.log_t(model_par = theta_try)
            if new_logp > self.logp_dead[it_num]:
                theta_new = theta_try
                break
        if theta_new is None:
            theta_new = self.sample_uniform_prior(num_samples = 1)
            # theta_new = sample_prior(rng, bounds, 1)[0]
        self.live_pts[self.worst_logp_index, :] = theta_new
        self.logp_live[self.worst_logp_index] = self.log_t(model_par = theta_new)
        # thetas[worst] = theta_new
        # logLs[worst] = log_likelihood(theta_new)
        
    
    def nested_sampling(self, n_live = 10, max_iter = 10, seed = 42,
                        max_up_attempts = 200):
        # Sample the prior and compute the logp of initial population
        np.random.seed(seed)
        self.live_pts, _, self.logp_live = self.brute_force_sampling(num_samples = n_live)
        self.init_pop = np.copy(self.live_pts)
        # Initialize variables
        self.dead_pts = np.zeros((max_iter, self.num_model_par))
        self.logp_dead = np.zeros(max_iter)
        self.logp_dead_weights = np.zeros(max_iter + n_live) # normalized
        # logp_current_it = np.amin(self.logp_live)
        self.delta_mu = np.zeros(max_iter)
        self.Xi = 1
        self.evidence = 0
        self.logZ = np.log(np.finfo(np.float64).eps)#-np.inf # Log evidence init
        self.info = 0.0 # information init
        self.logwidth = np.log(1.0 - np.exp(-1.0/n_live))  # initial shrinkage
        # init things that might not be used later (for checking)
        self.spread = np.zeros((max_iter, self.num_model_par))
        self.random_par_id = []
        self.worst_logp_index_list = []
        bar = tqdm(total = max_iter, desc='Nested sampling loop...', ascii=False)
        # print("All Logp {}".format(self.logp_live))
        for i in range(max_iter):
            # Find index of worst log-likelihood and update dead_pts and logp_dead
            self.get_worst_logp(it_num = i)
            self.worst_logp_index_list.append(self.worst_logp_index)
            # Update evidence Z using log-sum-exp for stability
            logZ_new = np.logaddexp(self.logZ, self.logwidth + self.logp_dead[i])
            delta_logZ = np.exp(self.logp_dead[i] + self.logwidth - logZ_new)
            self.info = delta_logZ * self.logp_dead[i] + (1 - delta_logZ) *\
                (self.info + self.logZ) - logZ_new
            self.logZ = logZ_new
            self.logwidth -= 1.0 / n_live
            self.delta_mu[i] = self.logwidth
            # Move                      
            # self.constrained_resample(i, max_up_attempts = max_up_attempts)
            # self.constrained_resample_rw(i, max_up_attempts = max_up_attempts)
            self.constrained_resample_slice(it_num = i, win_size = 0.4,
                                            max_up_attempts = max_up_attempts)
            bar.update(1)
        bar.close()
        log_weights_concat = np.concatenate((self.logp_dead, self.logp_live))
        self.weights = np.exp(log_weights_concat - np.max(log_weights_concat))
        self.weights /= np.sum(self.weights)
        self.prior_samples = np.concatenate((self.dead_pts, self.live_pts))
        # Final contribution of the live pts
        logL_max = np.max(self.logp_live)
        self.logZ = np.logaddexp(self.logZ, logL_max)
        
        
    # Make the move
    # print("Logp {:.2f} / Parameters: {}".format(self.logp_dead[i], 
    
    
    
    def nested_sampling2(self, n_live=200, max_iter=2000):
        
        # thetas = self.sample_uniform_prior(num_samples = n_live) #sample_prior(n_live)
        thetas, _, logLs = self.brute_force_sampling(num_samples = n_live) #logLs = np.array([loglike(th) for th in thetas])
        dead_points, dead_logLs, dead_logw = [], [], []
        logX_prev = 0.0 # Initialize logX
        Z = 0 # Initialize evidence
        bar = tqdm(total = max_iter,
                   desc='Nested sampling loop...', ascii=False)
        for i in range(max_iter):
            worst = np.argmin(logLs) # index of worst likelihood
            logL_star = logLs[worst] # worst likelihood value
            theta_star = thetas[worst] # worst theta value
            logX = -(i+1)/n_live # Current log X
            logw = np.log(np.exp(logX_prev)-np.exp(logX)) # Log of the weights (simple)
            dead_points.append(theta_star) # Append worst theta value
            dead_logLs.append(logL_star) # Append worst likelihood value
            dead_logw.append(logw) # Append weight value
            # Make the move of worst point
            seed = thetas[self.rng.integers(n_live)]
            new_theta, new_logL = self.constrained_move(seed, logL_star)
            # new_theta, new_logL = self.move_towards(theta_star, logL_star, 
            #                                         np.delete(thetas, worst, axis=0),
            #                                         dist_factor = 0.05)
            thetas[worst], logLs[worst] = new_theta, new_logL
            logX_prev = logX
            # Evidence update (Check it)
            # Z += np.exp(logX) * np.exp(logw) # Update evidence
            Z_dead = np.exp(logX) * np.exp(logw)
            Xm = np.exp(-i/n_live)
            Z_live = (Xm/n_live) * np.sum(np.exp(logLs))
            Z = Z + Z_dead + Z_live
            # Bar update
            bar.update(1)
        bar.close()
        # Posterior weights
        lw = np.array(dead_logLs) + np.array(dead_logw)
        # self.weights = np.exp(lw - np.max(lw))
        # self.weights /= np.sum(self.weights)
        # self.prior_samples = np.array(dead_points)
        log_dead_weights = lw
        log_live_weights = logLs
        log_weights_concat = np.concatenate((log_dead_weights, log_live_weights))
        self.weights = np.exp(log_weights_concat - np.max(log_weights_concat))
        self.weights /= np.sum(self.weights)
        self.prior_samples = np.concatenate((np.array(dead_points), thetas))
        return self.prior_samples, self.weights, log_weights_concat
    
    def get_destination(self, theta_start, theta_cluster, dist_factor = 0.2):
        # Mean of cluster
        cluster_mean = np.mean(theta_cluster, axis = 0)
        # Direction vector
        dir_vec = cluster_mean - theta_start
        # Compute total distance to mean
        distance = np.linalg.norm(dir_vec)
        # Make the move
        theta_dest = theta_start + dist_factor * distance * dir_vec/np.linalg.norm(dir_vec)
        return theta_dest
    
    def move_towards(self, theta_current, logL_current, theta_cluster,
                     dist_factor = 0.2, num_trials = 20):
        trial_num = 0
        logL_prop = np.log(np.finfo(float).eps)
        while trial_num <= num_trials or logL_prop < logL_current:
            theta_dest = self.get_destination(theta_current, theta_cluster,
                                              dist_factor = dist_factor)
            logL_prop = self.log_t(theta_dest)
            if logL_prop > logL_current:
                theta, logL = theta_dest, logL_prop
            trial_num += 1
        return theta, logL
    
    def constrained_move(self, theta0, logL_thresh, step=0.05, n_steps=50):
        theta = theta0.copy()
        logL = self.log_t(theta)
        for _ in range(n_steps):
            prop = theta + self.rng.normal(scale = step, size = self.num_model_par)
            # prop = self.sample_uniform_prior(num_samples = 1)[0,:]
            # reflect to stay inside bounds
            for j in range(self.num_model_par):
                if prop[j] < self.lower_bounds[j]:
                    prop[j] = self.lower_bounds[j] #2*self.lower_bounds[j] - prop[j]
                if prop[j] > self.upper_bounds[j]:
                    prop[j] = self.upper_bounds[j] #2*self.upper_bounds[j] - prop[j]
            logL_prop = self.log_t(prop)
            if logL_prop > logL_thresh:
                theta, logL = prop, logL_prop
        return theta, logL
    
    def constrained_move2(self, theta0, logL_thresh, num_trials = 20):
        theta = theta0.copy()
        logL = self.log_t(theta)
        trial_num = 0
        logL_prop = np.log(np.finfo(float).eps)
        while trial_num <= num_trials or logL_prop < logL_thresh:
            prop = prop = theta + self.rng.normal(scale = 0.05, size = self.num_model_par)#self.sample_uniform_prior(num_samples = 1)[0,:]
            for j in range(self.num_model_par):
                if prop[j] < self.lower_bounds[j]:
                    prop[j] = self.lower_bounds[j] + 0.0001 #2*self.lower_bounds[j] - prop[j]
                if prop[j] > self.upper_bounds[j]:
                    prop[j] = self.upper_bounds[j] - 0.0001 #2*self.upper_bounds[j] - prop[j]
            logL_prop = self.log_t(prop)
            if logL_prop > logL_thresh:
                theta, logL = prop, logL_prop
            trial_num += 1
        return theta, logL
        
    def weighted_mean(self, ):
        """ Weighted mean
        """
        self.mean_values = np.zeros(self.num_model_par)
        for jdim in range(self.num_model_par):
            self.mean_values[jdim] = np.sum(self.weights * self.prior_samples[:, jdim])
    
    def weighted_median(self, ):
        """ Weighted mean
        """
        self.median_values = self.weighted_quantile(quantile = 0.5)
    
    def weighted_var(self, ):
        """ Weighted variance and Standard deviation
        """
        self.var = np.zeros(self.num_model_par)
        for jdim in range(self.num_model_par):
            deviation = self.prior_samples[:, jdim] - self.mean_values[jdim]
            self.var[jdim] = np.sum(self.weights * deviation**2)
        self.std = np.sqrt(self.var)
    
    def ess_and_stderror(self, ):
        self.ess =  1.0 / np.sum(self.weights**2)
        self.standard_error = np.sqrt(self.var / self.ess)
        
    def confidence_interval(self, ci_percent = 95):
        """ Confidence interval with a given percentage
        """
        self.ci_percent = ci_percent
        lower_quantile = 0.5*(100-self.ci_percent)/100
        upper_quantile = (self.ci_percent + 0.5*(100-self.ci_percent))/100
        lower_bonds_ci = self.weighted_quantile(quantile = lower_quantile)
        upper_bonds_ci = self.weighted_quantile(quantile = upper_quantile)
        self.ci = np.vstack((lower_bonds_ci, upper_bonds_ci))
        
    def weighted_quantile(self, quantile = 0.5):
        posterior_quantile = np.zeros(self.num_model_par)
        for jdim in range(self.num_model_par):
            # Samples
            samples = self.prior_samples[:, jdim]
            # Sorted samples and weights
            idx = np.argsort(samples)
            x_sorted = samples[idx]
            w_sorted = self.weights[idx]
            # Cumulative distribution
            cdf = np.cumsum(w_sorted)
            # Interpolate cdf
            x_sorted_interp = np.linspace(x_sorted[0], x_sorted[-1], 
                                          num = 2*len(samples), endpoint=True)
            cdf_interpolated = np.interp(x_sorted_interp, x_sorted, cdf)
            # Find smallest j with cdf_j>= 𝑞uantile
            id_quantile = np.where(cdf_interpolated >= quantile)[0][0]
            posterior_quantile[jdim] = x_sorted_interp[id_quantile]
        return posterior_quantile
        # plt.plot(x_sorted_interp, cdf_interpolated)
        # plt.axvline(posterior_quantile[jdim])
           
    
    def compute_statistics(self, ):
        """ Compute statistical inferences
        """
        self.weighted_mean()
        self.weighted_median()
        self.weighted_var()
        self.ess_and_stderror()
        self.confidence_interval(ci_percent = 95)
    
    def plot_histograms(self, ):
        """ Plot histograms
        """
        plt.figure(figsize = (7,3))
        for jdim in range(self.prior_samples.shape[1]):
            plt.hist(self.prior_samples[:, jdim], bins = 100, weights = self.weights, 
                      density = True, label = r"$\theta_{}$".format(jdim), alpha = 0.7)
        plt.legend()
        plt.grid(linestyle = '--')
        plt.xlabel(r"Value")
        plt.ylabel(r"$p(\Theta)$")
        plt.tight_layout();
        
    def plot_smooth_marginal_posterior(self, figshape = None,
                                       figsize = None):
        """ Plot smooth posteriors using scipy KDE
        """
        if figshape is None:
            raise ValueError("I need a shape for the figure.")
        if figsize is None:
            figsize = (figshape[0]*4, figshape[1]*2)
           
        _, ax = self.give_me_an_ax(figshape = figshape, figsize = figsize)
        
        jdim = 0
        for row in range(ax.shape[0]):
            for col in range(ax.shape[1]):
                if jdim < self.prior_samples.shape[1]:
                    kde = scipy.stats.gaussian_kde(self.prior_samples[:, jdim], 
                                                   weights = self.weights)
                    x_grid = np.linspace(self.prior_samples[:, jdim].min(), 
                                         self.prior_samples[:, jdim].max(), 500)
                    pdf = kde(x_grid)
                    ax[row, col].plot(x_grid, pdf, linewidth = 1.5, 
                                      label = self.parameters_names[jdim], 
                                      alpha = 0.7, color = 'mediumpurple')
                    # ax[row, col].legend()
                    ax[row, col].grid(linestyle = '--')
                    ax[row, col].set_xlim((self.lower_bounds[jdim], self.upper_bounds[jdim]))
                    ax[row, col].set_xlabel(self.parameters_names[jdim])
                    ax[row, col].set_ylabel(r"$p(\theta)$")
                    jdim += 1
        plt.tight_layout();
        
        # plt.figure(figsize = (7,3))
        # for jdim in range(self.prior_samples.shape[1]):
        #     kde = scipy.stats.gaussian_kde(self.prior_samples[:, jdim], weights = self.weights)
        #     x_grid = np.linspace(self.prior_samples[:, jdim].min(), 
        #                          self.prior_samples[:, jdim].max(), 500)
        #     pdf = kde(x_grid)
        #     plt.plot(x_grid, pdf, linewidth = 2, label = r"$\theta_{}$".format(jdim), 
        #              alpha = 0.7)
        # plt.legend()
        # plt.grid(linestyle = '--')
        # plt.xlabel(r"$\theta$")
        # plt.ylabel(r"$p(\Theta)$")
        # plt.tight_layout();
    
    def plot_loglike_vs_mass(self, ax = None):
        """ Plots the evolution of the log-likelihood as a function of iteration number
        
        Parameters
        ----------
        ax : matplotlib axes or None
        """
        # Create axis if axis is None
        if ax is None:
            _, ax = self.give_me_an_ax()
        
        ax[0,0].plot(np.exp(self.delta_mu), self.logp_dead, '-k', linewidth = 1.5, alpha = 0.85)
        ax[0,0].grid(linestyle = '--')
        # ax.set_xlabel(r"iteration [-]")
        ax[0,0].set_xlabel(r"Prior mass [-]")
        ax[0,0].set_ylabel(r"$\mathcal{L}(\theta)$ [Np]")
        ax[0,0].set_xlim((0, 1))
        plt.tight_layout();
        
    def plot_loglike(self, ax = None):
        """ Plots the evolution of the log-likelihood as a function of iteration number
        
        Parameters
        ----------
        ax : matplotlib axes or None
        """
        # Create axis if axis is None
        if ax is None:
            _, ax = self.give_me_an_ax()
        
        ax[0,0].plot(self.logp_dead, '-k', linewidth = 1.5, alpha = 0.85)
        ax[0,0].grid(linestyle = '--')
        ax[0,0].set_xlabel(r"iteration [-]")
        ax[0,0].set_ylabel(r"$\mathcal{L}(\theta)$ [Np]")
        ax[0,0].set_xlim((0, len(self.logp_dead)))
        plt.tight_layout(); 
    
    def plot_spread(self, ax = None):
        """ Plots the spread as a function of iteration number
        
        Parameters
        ----------
        ax : matplotlib axes or None
        """
        # Create axis if axis is None
        if ax is None:
            _, ax = self.give_me_an_ax()
        for i in range(self.spread.shape[1]):
            ax[0,0].plot(self.spread[:, i], linewidth = 1, 
                    label = self.parameters_names[i] + \
                        r": ({}, {}). $s_o = {:.2f}$".format(self.lower_bounds[i], 
                                                             self.upper_bounds[i],
                                                             self.upper_bounds[i]-self.lower_bounds[i]), 
                    alpha = 0.85)
        ax[0,0].legend()
        ax[0,0].grid(linestyle = '--')
        ax[0,0].set_xlabel(r"iteration [-]")
        ax[0,0].set_ylabel(r"spread")
        ax[0,0].set_xlim((0, self.spread.shape[0]))
        # ax.set_ylim((np.amin(self.lower_bounds), np.amax(self.upper_bounds)))
        plt.tight_layout();
        
            
    def give_me_an_ax(self, figshape = (1, 1), figsize = (6,3)):
        """ return me a default matplotlib ax
        """
        fig, ax = plt.subplots(figsize = figsize,
                               nrows = figshape[0], ncols = figshape[1],
                               squeeze = False)
        return fig, ax
        
    # def plot_marginal_posterior(self, prior_samples, weights):
    #     """ Plot smooth posteriors
    #     """
    #     plt.figure(figsize = (7,3))
    #     for jdim in range(prior_samples.shape[1]):
    #         samples = prior_samples[:, jdim]
    #         idx = np.argsort(samples)
            
    #         plt.plot(samples[idx], weights[idx], linewidth = 2, label = r"$\theta_{}$".format(jdim), 
    #                  alpha = 0.7)
    #     plt.legend()
    #     plt.grid(linestyle = '--')
    #     plt.xlabel(r"$\theta$")
    #     plt.ylabel(r"$p(\Theta)$")
    #     plt.tight_layout();
        
    def reconstruct_mean(self, x_meas):
        """ Reconstruction from mean values
        """
        # prediction - from mean
        y_recon = self.model_fun(x_meas = x_meas,
                                 model_par = self.mean_values)
        return y_recon
    
    def reconstruct_ci(self, x_meas):
        """ Reconstruction from confidence interval values
        """
        # prediction - from mean
        y_recon = np.zeros((2, len(x_meas)))
        for jci in range(2):
            y_recon[jci, :] = self.model_fun(x_meas = x_meas,
                                             model_par = self.ci[jci, :])
        return y_recon
        
        
# returned_tuple =\
#     self.constrained_resample2(
#         current_worst_samplevec = self.dead_pts[i, :],
#         current_spread = current_spread,
#         current_worst_logp = self.logp_dead[i],
#         max_attempts = max_attempts)
# self.logp_live[self.worst_logp_index] = returned_tuple[0] 
# self.live_pts[self.worst_logp_index, :] = returned_tuple[1] 
# update_unsuccessful = returned_tuple[2]
# # 2 - If unsuccessful - select a random parameter as base and try again
# if update_unsuccessful:
#     all_indexes = np.arange(0, n_live)
#     remaining_indexes = np.delete(all_indexes, self.worst_logp_index)
#     random_index_id = np.random.randint(low = 0, 
#                                         high = len(remaining_indexes), 
#                                         size = 1)[0]
#     choosen_index = remaining_indexes[random_index_id]
#     returned_tuple =\
#         self.constrained_resample(
#             current_worst_samplevec = self.live_pts[choosen_index, :],
#             current_spread = current_spread,
#             current_worst_logp = self.logp_dead[i],
#             max_attempts = max_attempts)
#     self.logp_live[self.worst_logp_index] = returned_tuple[0] 
#     self.live_pts[self.worst_logp_index, :] = returned_tuple[1] 
#     update_unsuccessful = returned_tuple[2]
# # 3 - If unsuccessful - resample prior using the spread
# if update_unsuccessful:
#     mean = np.mean(self.live_pts, axis = 0)
#     prior_sample = np.random.uniform(low = mean-current_spread/2, 
#                                       high = mean+current_spread/2,
#                                       size = (1, self.num_model_par))[0]
#     # print(prior_sample)
#     logp_new = self.log_t(model_par = prior_sample)
#     if logp_new > self.logp_dead[i]:
#         self.logp_live[self.worst_logp_index] = logp_new 
#         self.live_pts[self.worst_logp_index, :] = np.copy(prior_sample) 
        #update_unsuccessful = False
    # else:
    #     logp_new = np.copy(current_worst_logp)
    #     suggested_parameter_vec = np.copy(current_worst_samplevec)        
    


