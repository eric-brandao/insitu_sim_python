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
try:
    import cvxpy as cvx
except:
    print("Not possible to use cvx")


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
        v = np.conjugate(v.T)
    else:
        v, sig, u = np.linalg.svd(np.conjugate(A.T), full_matrices=False)
        u = np.conjugate(u.T)
    return u, sig, v

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
    # b0 = (bm - (beta.T @ u).T)
    b0 = bm - u @ beta
    xi = np.divide(beta[0:int(p[0])], sig)
    # Call curvature calculator
    curv = curvature(reg_param, sig, beta, xi) # ok
    
    # Minimize 1
    curv_id = np.argmin(curv)
    x1 = reg_param[int(np.amin([curv_id+1, len(curv)-1]))]
    x2 = reg_param[int(np.amax([curv_id-1, 0]))]
    # print(x1)
    # print(x1.shape)
    # x1 = reg_param[int(np.amin([curv_id+1, len(curv)]))]
    # x2 = reg_param[int(np.amax([curv_id-1, 0]))]
    # Minimize 2 - set tolerance first (new versions of scipy need that)
    tolerance_array = np.zeros(len(x1)+len(x2)+1)
    tolerance_array[0:len(x1)] = x1.flatten()
    tolerance_array[len(x1):len(x1)+len(x2)] = x2.flatten()
    tolerance_array[-1] = 1e-5
    # print(tolerance_array)
    tolerance = np.amin(tolerance_array)#np.amin([x1/50, x2/50, 1e-5])
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
    return reg_c, reg_param, curv, rho_c, eta_c

def l_curve(u, sig, bm, plotit = False):
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
    lam_opt, reg_param, curv, rho_c, eta_c = l_corner(rho,eta,reg_param,u,sig,bm)
    # want to plot the L curve?
    if plotit:
        fig = plt.figure(figsize = (6,3))
        # fig.canvas.set_window_title("L-curve")
        plt.loglog(rho, eta)
        plt.loglog(rho_c, eta_c, '*r')
        plt.title('Reg. par: ' + "%.6f" % lam_opt)
        plt.xlabel(r'Residual norm $||Ax - b||_2$')
        plt.ylabel(r'Solution norm $||x||_2$')
        # plt.legend(loc = 'best')
        plt.grid(linestyle = '--', which='both')
        plt.tight_layout()
        ax2 = fig.add_axes([0.65, 0.58, 0.3, 0.3])
        ax2.semilogx(reg_param, -curv, 'k')
        ax2.semilogx(lam_opt, np.amax(-curv), '*r')
        ax2.axvline(lam_opt, color='grey',linestyle = '--', linewidth = 2, alpha = 0.4)
        ax2.grid()
        ax2.set_xlabel(r'$\lambda$')
        ax2.set_ylabel(r'$c(\lambda)$')
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
    
    # plt.figure(figsize = (6,4))
    # plt.loglog(reg_param , G)
    # minimization
    min_g = np.amin(G)
    min_g_id = np.where(G == min_g)[0][0]
    # print(min_g)
    # print(min_g_id)
    x1 = reg_param[int(np.amin(np.array([min_g_id + 1, npoints-1])))]
    x2 = reg_param[int(np.amax([min_g_id - 2, 0]))]
    # print(x1)
    # print(x2)

    # Minimize 2 - set tolerance first (new versions of scipy need that)
    tolerance = np.amin([x1/50, x2/50, 1e-5])
    reg_min = optimize.fminbound(gcvfun, x1, x2, 
                               args = (s2, beta[:p], delta0, True,  m-n), 
                               xtol=tolerance, full_output=False, disp=False)
    minG = gcvfun(reg_min, s2, beta[:p], delta0, True, m-n)
    # print(minG)
    #print(G)
    if print_gcvfun:
        plt.figure(figsize = (6,3))
        plt.loglog(reg_param , G)
        plt.loglog(reg_min, minG, '*r')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$G(\lambda)$')
        plt.title(r'GCV function: $\lambda = {}$'.format(reg_min))
        plt.grid()
        plt.tight_layout()
    
    return reg_min
        
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


def discrep(U, s, V, b, delta, x_0=None):
    m = U.shape[0]
    n = V.shape[0]
    p = len(s)
    ps = 1
    ld = 1
    x_delta = np.zeros((n, ld))
    lambda_val = np.zeros(ld)
    rho = np.zeros(p)
    
    if np.min(delta) < 0:
        raise ValueError("Illegal inequality constraint delta")
    
    if x_0 is None:
        x_0 = np.zeros(n)
    
    if ps == 1:
        omega = np.dot(V.T, x_0)
    else:
        omega = np.linalg.solve(V, x_0)
    
    beta = np.dot(U.T, b)
    delta_0 = np.linalg.norm(b - np.dot(U, beta))
    rho[p - 1] = delta_0 ** 2
    
    if ps == 1:
        for i in range(p - 1, 0, -1):
            rho[i - 1] = rho[i] + (beta[i] - s[i] * omega[i]) ** 2
    else:
        for i in range(0, p - 1):
            rho[i + 1] = rho[i] + (beta[i] - s[i, 0] * omega[i]) ** 2
    
    if np.min(delta) < delta_0:
        raise ValueError("Irrelevant delta < || (I - U*U'')*b ||")
    
    if ps == 1:
        s2 = s ** 2
        for k in range(ld):
            if delta ** 2 >= np.linalg.norm(beta - s * omega) ** 2 + delta_0 ** 2:
                x_delta[:, k] = x_0
            else:
                kmin = np.argmin(np.abs(rho - delta ** 2))
                lambda_0 = s[kmin]
                lambda_val[k] = newton(lambda_0, delta, s, beta, omega, delta_0)
                e = s / (s2 + lambda_val[k] ** 2)
                f = s * e
                x_delta[:, k] = np.dot(V[:, :p], e * beta + (1 - f) * omega)
    elif m >= n:
        omega = omega[:p]
        gamma = s[:, 0] / s[:, 1]
        x_u = np.dot(V[:, p:n], beta[p:n])
        for k in range(ld):
            if delta[k] ** 2 >= np.linalg.norm(beta[:p] - s[:, 0] * omega) ** 2 + delta_0 ** 2:
                x_delta[:, k] = np.dot(V, np.hstack((omega, np.dot(U[:, p:n].T, b))))
            else:
                kmin = np.argmin(np.abs(rho - delta[k] ** 2))
                lambda_0 = gamma[kmin]
                lambda_val[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma / (gamma ** 2 + lambda_val[k] ** 2)
                f = gamma * e
                x_delta[:, k] = np.dot(V[:, :p], (e * beta[:p] / s[:, 1]) + (1 - f) * s[:, 1] * omega) + x_u
    else:
        omega = omega[:p]
        gamma = s[:, 0] / s[:, 1]
        x_u = np.dot(V[:, p:m], beta[p:m])
        for k in range(ld):
            if delta[k] ** 2 >= np.linalg.norm(beta[:p] - s[:, 0] * omega) ** 2 + delta_0 ** 2:
                x_delta[:, k] = np.dot(V, np.hstack((omega, np.dot(U[:, p:m].T, b))))
            else:
                kmin = np.argmin(np.abs(rho - delta[k] ** 2))
                lambda_0 = gamma[kmin]
                lambda_val[k] = newton(lambda_0, delta[k], s, beta[:p], omega, delta_0)
                e = gamma / (gamma ** 2 + lambda_val[k] ** 2)
                f = gamma * e
                x_delta[:, k] = np.dot(V[:, :p], (e * beta[:p] / s[:, 1]) + (1 - f) * s[:, 1] * omega) + x_u
    
    return x_delta, lambda_val

def newton(lambda_0, delta, s, beta, omega, delta_0):
    thr = np.sqrt(np.finfo(float).eps)
    it_max = 50
    
    if lambda_0 < 0:
        raise ValueError("Initial guess lambda_0 must be nonnegative")
    
    p = len(s)
    ps = 1
    
    if ps == 2:
        sigma = s[:, 0]
        s = s[:, 0] / s[:, 1]
    
    s2 = s ** 2
    lambda_val = lambda_0
    step = 1
    it = 0
    
    while (abs(step) > thr * lambda_val and abs(step) > thr and it < it_max):
        it += 1
        f = s2 / (s2 + lambda_val ** 2)
        
        if ps == 1:
            r = (1 - f) * (beta - s * omega)
            z = f * r
        else:
            r = (1 - f) * (beta - sigma * omega)
            z = f * r
        
        step = (lambda_val / 4) * (np.dot(r.T, r) + (delta_0 + delta) * (delta_0 - delta)) / np.dot(z.T, r)
        lambda_val -= step
        
        if lambda_val < 0:
            lambda_val = 0.5 * lambda_0
            lambda_0 = 0.5 * lambda_0
    
    if abs(step) > thr * lambda_val and abs(step) > thr:
        raise ValueError("Max. number of iterations ({}) reached".format(it_max))
    
    return lambda_val


# from scipy.optimize import minimize_scalar

def ncp(U, s, b, printncp = False): #method='Tikh', 
    m = U.shape[0]
    p = len(s)
    ps = 1
    beta = np.dot(U.T, b)
    
    if ps == 2:
        s = s[p-1::-1, 0] / s[p-1::-1, 1]
        beta = beta[p-1::-1]
    
    npoints = 200
    nNCPs = 20
    smin_ratio = 16 * np.finfo(float).eps
    
    reg_param = np.zeros(npoints)
    reg_param[npoints - 1] = max([s[p - 1], s[0] * smin_ratio])
    ratio = (s[0] / reg_param[npoints - 1]) ** (1 / (npoints - 1))
    for i in range(npoints - 2, -1, -1):
        reg_param[i] = ratio * reg_param[i + 1]
    
    dists = np.zeros(npoints)
    if np.isrealobj(beta):
        q = m // 2
    else:
        q = m - 1
    cp = np.zeros((q, npoints))
    
    for i in range(npoints):
        # dists[i], cp[:, i] = ncpfun(reg_param[i], s, beta[:p], U[:, :p])
        dists[i] = ncpfun(reg_param[i], s, beta[:p], U[:, :p])
        cp[:, i], _ = ncpfun_cp(reg_param[i], s, beta[:p], U[:, :p])
    # print(cp)
    minGi = np.argmin(dists)
    # print(minGi)
    
    # minimization
    min_g = np.amin(dists)
    min_g_id = np.where(dists == min_g)[0][0]
    # print(min_g)
    # print(min_g_id)
    x1 = dists[int(np.amin(np.array([min_g_id, npoints])))]
    # x1 = dists[int(np.amin(np.array([min_g_id + 1, npoints])))]
    x2 = dists[int(np.amax([min_g_id - 2, 0]))]
    # print(x1)
    # print(x2)
    # reg_min_result = minimize_scalar(lambda reg: ncpfun(reg, s, beta[:p], U[:, :p]), 
    #                                   bounds=(x1, x2), 
    #                                   method='bounded')
    tolerance = np.amin([x1/50, x2/50, 1e-5])
    # print(U[:, :p])
    reg_min_result = optimize.fminbound(ncpfun, x1, x2, 
                                args = (s[:p], beta[:p], U[:, :p]), 
                                xtol=tolerance, full_output=False, disp=False)
    # print(reg_min_result)
    # reg_min = reg_min_result.x
    dist = ncpfun(reg_min_result, s[:p],  beta[:p], U[:, :p])
    cp_opt, cp_white =  ncpfun_cp(reg_min_result, s, beta[:p], U[:, :p])
    if printncp:
        stp = int(npoints/nNCPs)
        plt.figure(figsize = (6,4))
        plt.plot(cp[:,0:npoints:stp], '-.', linewidth = 0.5)
        plt.plot(cp_opt, '-r', linewidth = 2, label = r'optimal $c$')
        plt.plot(cp_white, '-k', linewidth = 2, label = r'white $c$')
        plt.legend()
        plt.grid()
        plt.xlabel('i')
        plt.ylabel(r'$c$')
        plt.title(r'$\lambda = {}$'.format(reg_min_result))
        plt.tight_layout()
    
    return reg_min_result#, dist


def ncpfun(lambda_val, s, beta, U, dsvd=False):
    if not dsvd:
        f = (lambda_val ** 2) / (s ** 2 + lambda_val ** 2)
    else:
        f = lambda_val / (s + lambda_val)
    
    r = np.dot(U, f * beta)
    m = len(r)
    
    if np.isrealobj(beta):
        q = m // 2
    else:
        q = m - 1
    
    D = np.abs(np.fft.fft(r)) ** 2
    D = D[1:q + 1]
    v = np.arange(1, q + 1) / q
    cp = np.cumsum(D) / np.sum(D)
    
    dist = np.linalg.norm(cp - v)
    # print(dist)
    return dist #, cp

def ncpfun_cp(lambda_val, s, beta, U, dsvd=False):
    if not dsvd:
        f = (lambda_val ** 2) / (s ** 2 + lambda_val ** 2)
    else:
        f = lambda_val / (s + lambda_val)
    r = np.dot(U, f * beta)
    m = len(r)
    if np.isrealobj(beta):
        q = m // 2
    else:
        q = m - 1
    D = np.abs(np.fft.fft(r)) ** 2
    D = D[1:q + 1]
    v = np.arange(1, q + 1) / q  
    cp = np.cumsum(D) / np.sum(D)
    #dist = np.linalg.norm(cp - v)
    return cp, v


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
    # H2 = np.zeros((2*h_mtx.shape[0], 2*h_mtx.shape[1]))
    # H2[0:h_mtx.shape[0], 0:h_mtx.shape[1]] = h_mtx.real
    # H2[h_mtx.shape[0]:, 0:h_mtx.shape[1]] = -h_mtx.imag
    # H2[0:h_mtx.shape[0], h_mtx.shape[1]:] = h_mtx.imag
    # H2[h_mtx.shape[0]:, h_mtx.shape[1]:] = h_mtx.real
    
    # p2 = np.zeros(2*len(bm))
    # p2[0:len(bm)] = bm.real
    # p2[len(bm):] = bm.imag
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
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

def least_sq_solver(h_mtx, bm):
    """ least squares solver
    """
    x_lsq = np.linalg.lstsq(h_mtx, bm)[0]
    return x_lsq

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
    #constraints = [cvx.pnorm(b - cvx.matmul(A, x), p=2) <= noise_norm]
    constraints = [cvx.pnorm(A @ x - b, p = 2) <= noise_norm]
    
    # Form objective.
    obj = cvx.Minimize(cvx.norm(x, l_norm))
    
    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    prob.solve();
    return x.value

def cvx_solver_c(A, b, noise_norm, l_norm = 2):
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
    x = cvx.Variable(shape = l, complex = True, value = np.zeros(l))
    
    # Create constraint.
    #constraints = [cvx.norm(A @ x - b, 2) <= noise_norm]
    constraints = [cvx.pnorm(b - cvx.matmul(A, x), p=2) <= noise_norm]
    # Form objective.
    obj = cvx.Minimize(cvx.pnorm(x, p = l_norm))
    
    # Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    prob.solve();
    return x.value

def cvx_tikhonov(A, b, lam, l_norm = 2):
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
    x = cvx.Variable(shape = l, complex = True)
    
  
    # Form objective.
    obj = cvx.Minimize(cvx.norm(A @ x - b, 2) + (lam)*cvx.norm(x, l_norm))
    
    # Form and solve problem.
    prob = cvx.Problem(obj)
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
        b : numpy 1darray
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
    minval = np.amin([0.1*s[-1], 0.1*np.finfo(float).eps*s[0]])
    plt.ylim((minval, 100*s[0]))
    plt.xlabel(r'$i$')
    plt.ylabel(r'$\sigma_i$, $|U^Tb|$, $|U^Tb|/\sigma_i$')
    plt.title('cond(A) = {0:.2f}'.format(cond_number), loc='right')
    plt.grid()
    plt.tight_layout()
    
def nmse(x_sol, x_truth):
    """ returns the NMSE (normalized mean squared error)

    Parameters
    ----------
        x_sol : numpy 1darray
            solution
        x_sol : numpy 1darray
            ground truth
    Returns
    -------
        nnse : float
            estimated NMSE
    """
    nmse = (np.linalg.norm(x_sol-x_truth)/np.linalg.norm(x_truth))**2
    return nmse

def mae(x_sol, x_truth):
    """ returns the MAE (nmean absolute error)

    Parameters
    ----------
        x_sol : numpy 1darray
            solution
        x_sol : numpy 1darray
            ground truth
    Returns
    -------
        mae : float
            estimated MAE
    """
    mae = np.linalg.norm(x_sol-x_truth)
    return mae

def nmse_freq(x_sol, x_truth):
    """ returns the NMSE vs freq (normalized mean squared error)

    Parameters
    ----------
        x_sol : numpy ndarray
            solution arraged in Nvals x Nfreq 
        x_sol : numpy ndarray
            ground truth arraged in Nvals x Nfreq 
    Returns
    -------
        nnse : nympy 1dArray
            estimated NMSE vs freq
    """
    _, nfreq = x_sol.shape
    nmse_freq = np.zeros(nfreq)
    for jf in np.arange(nfreq):
        nmse_freq[jf] = nmse(x_sol[:,jf], x_truth[:,jf])
    return nmse_freq