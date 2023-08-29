""" L-curve regularization parameter finding.

This set of functions are used to find an optimal regularization parameter
for a under determined system of equations. The optimal parameter is found
according to the L-curve criteria, described in

Discrete Inverse Problems - insight and algorithms, Per Christian Hansen,
Technical University of Denmark, DTU compute, 2010

The code was adapted from http://www.imm.dtu.dk/~pcha/Regutools/ by
Per Christian Hansen, DTU Compute, October 27, 2010 - originally implemented in Matlab

This version is only for the under determined case.
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import linalg # for svd
from sklearn.linear_model import Ridge
from scipy import optimize
import warnings
import cvxpy as cvx

def curvature(lambd, sig, beta, xi):
    """ computes the NEGATIVE of the curvature.

    Parameters
    ----------
        lambd : float
            regularization parameter
        sig : numpy 1darray
            singular values
        beta: numpy 1darray
            conj(u) @ bm
        xi : numpy 1darray
            beta / sig
    Returns
    -------
        curv : numpy 1darray
            negative curvature
    """
    # Initialization.
    phi = np.zeros(lambd.shape)
    dphi = np.zeros(lambd.shape)
    psi = np.zeros(lambd.shape)
    dpsi = np.zeros(lambd.shape)
    eta = np.zeros(lambd.shape)
    rho = np.zeros(lambd.shape)
    if len(beta) > len(sig): # A possible least squares residual.
        LS = True
        rhoLS2 = beta[-1] ** 2
        beta = beta[0:-2]
    else:
        LS = False
    # Compute some intermediate quantities.
    for jl, lam in enumerate(lambd):
        f  = np.divide((sig ** 2), (sig ** 2 + lam ** 2)) # ok
        cf = 1 - f # ok
        eta[jl] = np.linalg.norm(f * xi) # ok
        rho[jl] = np.linalg.norm(cf * beta)
        f1 = -2 * f * cf / lam 
        f2 = -f1 * (3 - 4*f)/lam
        phi[jl]  = np.sum(f*f1*np.abs(xi)**2) #ok
        psi[jl] = np.sum(cf*f1*np.abs(beta)**2)
        dphi[jl] = np.sum((f1**2 + f*f2)*np.abs(xi)**2)
        dpsi[jl] = np.sum((-f1**2 + cf*f2)*np.abs(beta)**2) #ok

    if LS: # Take care of a possible least squares residual.
        rho = np.sqrt(rho ** 2 + rhoLS2)

    # Now compute the first and second derivatives of eta and rho
    # with respect to lambda;
    deta  =  np.divide(phi, eta) #ok
    drho  = -np.divide(psi, rho)
    ddeta =  np.divide(dphi, eta) - deta * np.divide(deta, eta)
    ddrho = -np.divide(dpsi, rho) - drho * np.divide(drho, rho)

    # Convert to derivatives of log(eta) and log(rho).
    dlogeta  = np.divide(deta, eta)
    dlogrho  = np.divide(drho, rho)
    ddlogeta = np.divide(ddeta, eta) - (dlogeta)**2
    ddlogrho = np.divide(ddrho, rho) - (dlogrho)**2
    # curvature.
    curv = - np.divide((dlogrho * ddlogeta - ddlogrho * dlogeta),
        (dlogrho**2 + dlogeta**2)**(1.5))
    return curv

def l_corner(rho,eta,reg_param,u,sig,bm):
    """ Computes the corner of the L-curve.

    Uses the function "curvature"

    Parameters
    ----------
        rho : numpy 1darray
            computed in l_curve function (residual norm) - related to curvature
        eta : numpy 1darray
            computed in l_curve function (solution norm) - related to curvature
        reg_param : numpy 1darray
            computed in l_curve function
        u : numpy ndarray
            left singular vectors
        sig : numpy 1darray
            singular values
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
    Returns
    -------
        reg_c : float
            optimal regularization parameter
    """
    # Set threshold for skipping very small singular values in the analysis of a discrete L-curve.
    s_thr = np.finfo(float).eps # Neglect singular values less than s_thr.
    # Set default parameters for treatment of discrete L-curve.
    deg   = 2  # Degree of local smooting polynomial.
    q     = 2  # Half-width of local smoothing interval.
    order = 4  # Order of fitting 2-D spline curve.
    # Initialization.
    if (len(rho) < order):
        print('I will fail. Too few data points for L-curve analysis')
    Nm, Nu = u.shape
    p = sig.shape
    beta = (np.conj(u).T) @ bm 
    beta = np.reshape(beta[0:int(p[0])], beta.shape[0])
    b0 = (bm - (beta.T @ u).T)
    xi = np.divide(beta[0:int(p[0])], sig)
    # Call curvature calculator
    curv = curvature(reg_param, sig, beta, xi) # ok
    # Minimize 1
    curv_id = np.argmin(curv)
    x1 = reg_param[int(np.amin([curv_id+1, len(curv)-1]))]
    x2 = reg_param[int(np.amax([curv_id-1, 0]))]
    # x1 = reg_param[int(np.amin([curv_id+1, len(curv)]))]
    # x2 = reg_param[int(np.amax([curv_id-1, 0]))]
    # Minimize 2 - set tolerance first (new versions of scipy need that)
    tolerance = np.amin([x1/50, x2/50, 1e-5])
    reg_c = optimize.fminbound(curvature, x1, x2, args = (sig, beta, xi), xtol=tolerance,
        full_output=False, disp=False)
    kappa_max = - curvature(reg_c, sig, beta, xi) # Maximum curvature.
    if kappa_max < 0:
        lr = len(rho)
        reg_c = reg_param[lr-1]
        rho_c = rho[lr-1]
        eta_c = eta[lr-1]
    else:
        f = np.divide((sig**2), (sig**2 + reg_c**2))
        eta_c = np.linalg.norm(f * xi)
        rho_c = np.linalg.norm((1-f) * beta[0:len(f)])
        if Nm > Nu:
            rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0)**2)
    return reg_c

def csvd(A):
    """ Computes the SVD based on the size of A.

    Parameters
    ----------
        A : numpy ndarray
            sensing matrix (Nm x Nu). Nm are the number of measurements
            and Nu the number of unknowns
    Returns
    -------
        u : numpy ndarray
            left singular vectors
        sig : numpy 1darray
            singular values
        v : numpy ndarray
            right singular vectors
    """
    Nm, Nu = A.shape
    if Nm >= Nu: # more measurements than unknowns
        u, sig, v = np.linalg.svd(A, full_matrices=False)
    else:
        v, sig, u = np.linalg.svd(np.conjugate(A.T), full_matrices=False)
    return u, sig, v

def l_cuve(u, sig, bm, plotit = False):
    """ Find the optimal regularizatin parameter.

    This function uses the L-curve and computes its curvature in
    order to find its corner - optimal regularization parameter.

    Uses the function "l_corner"

    Parameters
    ----------
        u : numpy ndarray
            left singular vectors
        sig : numpy 1darray
            singular values
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
        plotit : bool
            whether to plot the L curve or not. Default is False
    Returns
    -------
        lam_opt : float
            optimal regularization parameter
    """
    # Set defaults.
    npoints = 200  # Number of points on the L-curve
    smin_ratio = 16*np.finfo(float).eps  # Smallest regularization parameter.
    # Initialization.
    Nm, Nu = u.shape
    p = sig.shape
    beta = np.conjugate(u).T @ bm
    beta2 = np.linalg.norm(bm) ** 2 - np.linalg.norm(beta)**2
    s = sig
    beta = np.reshape(beta[0:int(p[0])], beta.shape[0])
    xi = np.divide(beta[0:int(p[0])],s)
    xi[np.isinf(xi)] = 0

    eta = np.zeros((npoints,1))
    rho = np.zeros((npoints,1)) #eta
    reg_param = np.zeros((npoints,1))
    s2 = s ** 2
    reg_param[-1] = np.amax([s[-1], s[0]*smin_ratio])
    ratio = (s[0]/reg_param[-1]) ** (1/(npoints-1))
    for i in np.arange(start=npoints-2, step=-1, stop = -1):
        reg_param[i] = ratio*reg_param[i+1]
    for i in np.arange(start=0, step=1, stop = npoints):
        f = s2 / (s2 + reg_param[i] ** 2)
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm((1-f) * beta[:int(p[0])])
    if (Nm > Nu and beta2 > 0):
        rho = np.sqrt(rho ** 2 + beta2)
    # Compute the corner of the L-curve (optimal regularization parameter)
    lam_opt = l_corner(rho,eta,reg_param,u,sig,bm)
    # want to plot the L curve?
    if plotit:
        fig = plt.figure()
        # fig.canvas.set_window_title("L-curve")
        plt.loglog(rho, eta, label='Reg. par: ' + "%.6f" % lam_opt)
        plt.xlabel(r'Residual norm $||Ax - b||_2$')
        plt.ylabel(r'Solution norm $||x||_2$')
        plt.legend(loc = 'best')
        plt.grid(linestyle = '--', which='both')
        plt.tight_layout()
    return lam_opt

def gcv_lambda(u,s,bm, print_gcvfun = False):
    """ Estimates the optimal regularization parameter via Generalized Cross val.
    
    Parameters
    ----------
        u : numpy ndarray
            left singular vectors from csvd
        sig : numpy 1darray
            singular values from csvd
        bm : numpy 1darray
            measured vector
    Returns
    -------
        lambda : float
            estimated regularization parameter
    """
    # Set defaults.
    npoints = 200  # Number of points on the L-curve
    smin_ratio = 16*np.finfo(float).eps  # Smallest regularization parameter.
    # Initialization.
    m, n = u.shape
    p = len(s)
    beta = np.conjugate(u).T @ bm
    #print(p)
    #print(beta.shape)
    beta2 = np.linalg.norm(bm) ** 2 - np.linalg.norm(beta)**2
    #print(beta2)
    # Vector of regularization parameters.
    reg_param = np.zeros(npoints)
    G = np.zeros(npoints)
    s2 = s**2
    reg_param[-1] = np.amax([s[-1], s[0]*smin_ratio])
    ratio = (s[0]/reg_param[-1])**(1/(npoints-1))
    #print(ratio)
    for i in np.arange(start=npoints-2, step=-1, stop = -1):
        reg_param[i] = ratio*reg_param[i+1]
    #print(reg_param)
    # Intrinsic residual.
    delta0 = 0
    if (m > n and beta2 > 0):
        delta0 = beta2
    # Vector of GCV-function values.
    for i in np.arange(npoints):
        G[i] = gcvfun(reg_param[i], s2, beta[:p], delta0, 
                      compute_mn = True, mn = m-n)
    #print(G)
    if print_gcvfun:
        plt.figure(figsize = (6,4))
        plt.loglog(reg_param , G)
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$G(\lambda)$')
        plt.title('GCV function')
        plt.grid()
        plt.tight_layout()
        
def gcvfun(lam, s2, beta, delta0, compute_mn = True, mn = 0):
    """ Auxiliary routine for gcv.  PCH, IMM, Feb. 24, 2008.

        Note: f = 1 - filter-factors.
    """
    if compute_mn:
        f = (lam ** 2)/(s2 + lam**2)
    else:
       f = lam/(s2 + lam)
    G = (np.linalg.norm(f * beta)**2 + delta0)/(mn + np.sum(f))**2
    return G

def tikhonov(u,s,v,b,lambd_value):
    """ Tikhonov regularization. Needs some work

    Computes the Tikhonov regularized solution x_lambda, given the SVD or
    GSVD as computed via csvd or cgsvd, respectively.  The SVD is used,
    i.e. if U, s, and V are specified, then standard-form regularization
    is applied:
    min { || A x - b ||^2 + lambda^2 || x - x_0 ||^2 } .
    Valid for underdetermined systems.
    Based on the matlab routine by: Per Christian Hansen, DTU Compute, April 14, 2003.
    Reference: A. N. Tikhonov & V. Y. Arsenin, "Solutions of Ill-Posed
    Problems", Wiley, 1977.

    Parameters
    ----------
        u : numpy ndarray
            left singular vectors
        sig : numpy 1darray
            singular values
        v : numpy ndarray
            right singular vectors
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
        lambd_value : float
            optimal regularization parameter
    Returns
    -------
        x_lambda : numpy 1darray
            estimated solution to inverse problem
    """
    # warn that lambda should be bigger than 0
    if lambd_value < 0:
        warnings.warn("Illegal regularization parameter lambda. I'll set it to 1.0")
        lambd_value = 1.0
    # m = u.shape[0]
    # n = v.shape[0]
    p = len(s)
    # ps = 1
    beta = np.conjugate(u[:,0:p]).T @ b
    zeta = s * beta
    # ll = length(lambda); x_lambda = zeros(n,ll);
    # rho = zeros(ll,1); eta = zeros(ll,1);
    # The standard-form case.
    x_lambda = v[:,0:p] @ np.divide(zeta, s**2 + lambd_value**2)
    
    # because csvd takes the hermitian of h_mtx and only the first m collumns of v
    # phi_factors = (s**2)/(s**2+lambd_value**2)
    # x = (v @ np.diag(phi_factors/s) @ np.conjugate(u)) @ b
    # beta_try = np.conjugate(u) @ b
    # zeta_try = s*beta_try
    # x_try = v @ np.divide(zeta_try, s**2 + lambd_value**2) #np.diag(s/(s**2+lambd_value**2)) @ beta_try
    return x_lambda


def ridge_solver(h_mtx,bm,lambd_value):
    """ Ridge regression. 

    Parameters
    ----------
        h_mtx : numpy ndarray
            sensing matrix
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
        lambd_value : float
            optimal regularization parameter
    Returns
    -------
        x_lambda : numpy 1darray
            estimated solution to inverse problem
    """
    # Form a real H2 matrix and p2 measurement
    H2 = np.vstack((np.hstack((h_mtx.real, -h_mtx.imag)),
        np.hstack((h_mtx.imag, h_mtx.real))))
    p2 = np.vstack((bm.real,bm.imag)).flatten()
    regressor = Ridge(alpha=lambd_value, fit_intercept = False, solver = 'svd')
    x2 = regressor.fit(H2, p2).coef_
    x_lambda = x2[:h_mtx.shape[1]]+1j*x2[h_mtx.shape[1]:]
    return x_lambda


def direct_solver(h_mtx,bm,lambd_value):
    """ Solves the Tikhonov regularization with analytical sol.

    Parameters
    ----------
        h_mtx : numpy ndarray
            sensing matrix
        bm: numpy 1darray
            your measurement vector (size: Nm x 1)
        lambd_value : float
            optimal regularization parameter
    Returns
    -------
        x_lambda : numpy 1darray
            estimated solution to inverse problem
    """
    Hm = np.matrix(h_mtx)
    x_lambda = Hm.getH() @ np.linalg.inv(Hm @ Hm.getH() +\
                                         (lambd_value**2)*np.identity(len(bm))) @ bm
    return x_lambda   

# np.random.seed(0)
# H = np.random.randn(7, 10)
# p = np.random.randn(7)
# u, sig, v = csvd(H)
# lambd_value = l_cuve(u, sig, p, plotit=False)
# tikhonov(u, sig, v, p, lambd_value)
# H = np.array([[1, 7, 10],[2.3, 5.4, 13.2]])
# p = np.array([1.4, 8.54])
# u, sig, v = csvd(H)
# x_lambda = tikhonov(u, sig, v, p, 0.3)
# print(x_lambda)

def cvx_solver(A, b, noise_norm, l_norm = 2):
    """ Solves regularized problem by convex optmization.

    Parameters
    ----------
        A : numpy ndarray
            sensing matrix (MxL)
        b: numpy 1darray
            your measurement vector (size: M x 1)
        noise_norm : float
            norm of the noise (to set constraint)
        l_norm : int
            Type of norm to minimize x
    Returns
    -------
        x : numpy 1darray
            estimated solution to inverse problem
    """
    # Create variable to be solved for.
    m, l = A.shape
    x = cvx.Variable(shape = l)
    
    # Create constraint.
    constraints = [cvx.norm(A @ x - b, 2) <= noise_norm]
    
    # Form objective.
    obj = cvx.Minimize(cvx.norm(x, l_norm))
    
    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    prob.solve();
    return x.value

def tsvd(u,s,v,b,k):
    """ Estimates truncated SVD regularized solution
    
    Parameters
    ----------
        u : numpy ndarray
            left singular vectors from csvd
        s : numpy 1darray
            singular values from csvd
        v : numpy ndarray
            right singular vectors from csvd
        s : numpy 1darray
            measured vector
        k : int
            number of singular values to include
    Returns
    -------
        x_k : numpy 1darray
            estimated solution to inverse problem
    """
    n,p = v.shape
    #lk = length(k);
    if k > p:
      warnings.warn('Illegal truncation parameter k. Setting k = p')
      k = p
    
    #eta = zeros(lk,1); rho = zeros(lk,1);
    beta = np.conj(u[:,0:p]).T @ b
    xi = beta/s
    v = np.conj(v).T    
    x_k = v[:,0:k] @ xi[0:k]
    return x_k

def ssvd(u,s,v,b,tau):
    """ Estimates selective SVD regularized solution
    
    Parameters
    ----------
        u : numpy ndarray
            left singular vectors from csvd
        s : numpy 1darray
            singular values from csvd
        v : numpy ndarray
            right singular vectors from csvd
        s : numpy 1darray
            measured vector
        tau : float
            Threshhold
    Returns
    -------
        x_k : numpy 1darray
            estimated solution to inverse problem
    """
    n,p = v.shape
        
    beta_full = np.conj(u).T @ b
    idbeta = np.where(np.abs(beta_full) > tau)[0]
    beta = beta_full[idbeta]
    xi = beta/s[idbeta]
    v = np.conj(v).T
    v = v[:,idbeta]    
    x_tau = v @ xi
    return x_tau


    

def plot_colvecs(U, rows = 4, cols = 4, figsize = (8,5), ylim = (-0.2,0.2)):
    """ Plot the column vectors
    """
    fig, axs = plt.subplots(rows, cols, figsize = (8,5), sharex=True, sharey=True)
    j = 0
    for row in np.arange(rows):
        for col in np.arange(cols):
            axs[row,col].plot(U[:,j])
            axs[row,col].set_ylim(ylim)
            j += 1
            axs[rows-1,col].set_xlabel(r'$i$')
        axs[row,0].set_ylabel(r'$u$')
    plt.tight_layout()
    
def plot_picard(U,s,b, noise_norm = None,figsize = (5,4)):
    """ Make Picard plot
    """
    # condition number
    cond_number = s[0]/s[-1]
    
    # beta
    beta = np.abs(U.T @ b)
    
    # Figure
    plt.figure(figsize = figsize)
    plt.semilogy(np.abs(s), '+k', label = r'$\sigma$')
    plt.semilogy(beta, 'xr', label = r'$|U^T b|$')
    plt.semilogy(beta/s, '.b', label = r'$|U^T b|/\sigma$')
    plt.semilogy(np.finfo(float).eps*s[0]*np.ones(len(s)), '--', linewidth = 2, 
                 color = 'Grey', label = r'eps$\cdot \sigma_1$')
    if noise_norm is not None:
        plt.semilogy((noise_norm/np.linalg.norm(b))*s[0]*np.ones(len(s)), '--', linewidth = 2, 
                     color = 'Grey', 
                     label = r'$\left\|n\right\|_2/\left\|b\right\|_2 \cdot \sigma_1$')
    plt.legend(loc = 'lower left')
    plt.ylim((0.1*s[-1], 100*s[0]))
    plt.xlabel(r'$i$')
    plt.ylabel(r'$\sigma_i$, $|U^Tb|$, $|U^Tb|/\sigma_i$')
    plt.title('cond(A) = {0:.2f}'.format(cond_number), loc='right')
    plt.grid()
    
def nmse(x_sol, x_truth):
    nmse = (np.linalg.norm(x_sol-x_truth)/np.linalg.norm(x_truth))**2
    return nmse