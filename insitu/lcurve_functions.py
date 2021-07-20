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
from scipy import optimize
import warnings

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
    beta = (np.conj(u)) @ bm 
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
    beta = np.conjugate(u) @ bm
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
        fig.canvas.set_window_title("L-curve")
        plt.loglog(rho, eta, label='Reg. par: ' + "%.6f" % lam_opt)
        plt.xlabel(r'Residual norm $||Ax - b||_2$')
        plt.ylabel(r'Solution norm $||x||_2$')
        plt.legend(loc = 'best')
        plt.grid(linestyle = '--', which='both')
        plt.tight_layout()
    return lam_opt

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
        lam_opt : float
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

    