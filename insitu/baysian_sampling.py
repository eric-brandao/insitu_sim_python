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
                 num_model_par = 1, seed = 42):
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
        else:
            self.measured_coords = measured_coords
            self.measured_data = measured_data
            self.num_of_meas = len(measured_data)
            self.num_model_par = num_model_par
            self.seed = seed
            self.rng = np.random.default_rng(seed)
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
        # error norm
        error_norm = np.linalg.norm(self.measured_data - y_pred)
        logp = -(self.num_of_meas/2)*np.log((error_norm**2)/2)
        return logp
    
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
    
    
    def nested_sampling(self, n_live=200, max_iter=2000):
        
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
            # seed = thetas[self.rng.integers(n_live)]
            # new_theta, new_logL = self.constrained_move2(seed, logL_star)
            new_theta, new_logL = self.move_towards(theta_star, logL_star, 
                                                    np.delete(thetas, worst, axis=0),
                                                    dist_factor = 0.05)
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
        return self.prior_samples, self.weights, Z
    
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
        
    def plot_smooth_marginal_posterior(self, ):
        """ Plot smooth posteriors using scipy KDE
        """
        plt.figure(figsize = (7,3))
        for jdim in range(self.prior_samples.shape[1]):
            kde = scipy.stats.gaussian_kde(self.prior_samples[:, jdim], weights = self.weights)
            x_grid = np.linspace(self.prior_samples[:, jdim].min(), 
                                 self.prior_samples[:, jdim].max(), 500)
            pdf = kde(x_grid)
            plt.plot(x_grid, pdf, linewidth = 2, label = r"$\theta_{}$".format(jdim), 
                     alpha = 0.7)
        plt.legend()
        plt.grid(linestyle = '--')
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$p(\Theta)$")
        plt.tight_layout();
        
    def plot_marginal_posterior(self, prior_samples, weights):
        """ Plot smooth posteriors
        """
        plt.figure(figsize = (7,3))
        for jdim in range(prior_samples.shape[1]):
            samples = prior_samples[:, jdim]
            idx = np.argsort(samples)
            
            plt.plot(samples[idx], weights[idx], linewidth = 2, label = r"$\theta_{}$".format(jdim), 
                     alpha = 0.7)
        plt.legend()
        plt.grid(linestyle = '--')
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$p(\Theta)$")
        plt.tight_layout();
        
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
        
        
        
    


