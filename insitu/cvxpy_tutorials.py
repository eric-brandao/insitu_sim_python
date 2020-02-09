#%%
# import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import linalg # for svd
from lcurve_functions import csvd, l_cuve

#%%
mat = scio.loadmat('c0.mat')
c0 = np.array(mat['c0'])[0][0]

mat = scio.loadmat('freq.mat')
ind_f = 1
freq = np.array(mat['freq_m'])[0][ind_f]
print('Running test for freq: {}'.format(freq))
k0 = 2 * np.pi * freq / c0

mat = scio.loadmat('receivers.mat')
receivers = np.array(mat['receivers_m'])

mat = scio.loadmat('directions.mat')
directions = np.array(mat['dir_m'])
k_vec = k0 * directions

mat = scio.loadmat('pm.mat')
pm_all = np.array(mat['p_m'])
pm = pm_all[:,:,ind_f]
pm = pm.T

h_mtx = np.exp(1j*receivers @ k_vec.T)
print('the shape of H is {}'.format(h_mtx.shape))

##### %% SVD ####
u, sig, v = csvd(h_mtx)
# u, s, v = linalg.svd(h_mtx, full_matrices=False) #compute SVD without 0 singular values
# print('singular values of shape {} and value {}'.format(sig.shape, sig[0:6]))
# print('u of shape {} and value {}'.format(u.shape, u[2, 2]))
# print('v of shape {} and value {}'.format(v.shape, v[2, 2]))
lam_opt = l_cuve(u, sig, pm, plotit=False)
print('Optmal regu par is: {}'.format(lam_opt))
#%% Least squares problems with no contraints
# Generate data.
# m = 20
# n = 15
# np.random.seed(1)
# A = np.random.randn(m, n) + 1j*np.random.randn(m, n)
# b = np.random.randn(m) + 1j*np.random.randn(m)

# # Define and solve the CVXPY problem.
# x = cp.Variable(n, complex = True)
# cost = cp.sum_squares(A*x - b)
# prob = cp.Problem(cp.Minimize(cost))
# prob.solve()

# # Print result.
# print("\nThe optimal value is", prob.value)
# print("The optimal x is")
# print(x.value)
# print("The norm of the residual is ", cp.norm(A*x - b, p=2).value)

#%% Ridge regression with cvxpy
# def loss_fn(H, pm, x):
#     return cp.pnorm(cp.matmul(H, x) - pm, p=2)**2

# def regularizer(x):
#     return cp.pnorm(x, p=2)**2

# def objective_fn(H, pm, x, lambd):
#     return loss_fn(H, pm, x) + lambd * regularizer(x)

# def mse(H, pm, x):
#     return (1.0 / H.shape[0]) * loss_fn(H, pm, x).value

# def generate_data(m=100, n=20, sigma=5):
#     "Generates data matrix X and observations Y."
#     np.random.seed(1)
#     x_star = np.random.randn(n)
#     # Generate an ill-conditioned data matrix
#     H = np.random.randn(m, n)
#     # Corrupt the observations with additive Gaussian noise
#     pm = H.dot(x_star) + np.random.normal(0, sigma, size=m)
#     return H, pm

# ## Generate the data
# m = 100
# n = 20
# sigma = 5

# H, pm = generate_data(m, n, sigma)
# H_train = H[:50, :] # first fifty
# pm_train = pm[:50]
# H_test = H[50:, :] # last fifity
# pm_test = pm[50:]

# ## write the problem
# x = cp.Variable(n)
# lambd = cp.Parameter(nonneg=True)
# problem = cp.Problem(cp.Minimize(objective_fn(H_train, pm_train, x, lambd)))
# lambd_values = np.logspace(-5, 1, 50)
# print(lambd_values)
# train_errors = []
# test_errors = []
# x_values = []
# solution_norm = []
# residual_norm = []
# for v in lambd_values:
#     lambd.value = v
#     problem.solve()
#     train_errors.append(mse(H_train, pm_train, x))
#     test_errors.append(mse(H_test, pm_test, x))
#     x_values.append(x.value)
#     solution_norm.append(np.sum((np.abs(x.value))**2)) #+= [(lm.coef_**2).sum()]
#     residual_norm.append(np.sum((np.abs(H_train @ x.value - pm_train))**2)) #+= [((lm.predict(H) - pm)**2).sum()]

# # %matplotlib inline
# # %config InlineBackend.figure_format = 'svg'
# def plot_lcurve(solution_norm, residual_norm):
#     # plt.plot(np.log(residual_norm), np.log(solution_norm))
#     plt.plot(residual_norm, solution_norm)
#     plt.xscale("log")
#     plt.yscale("log")
#     # plt.legend(loc="upper left")
#     plt.xlabel('|Hx-p|', fontsize=16)
#     plt.ylabel("|x|")
#     plt.show()

# plot_lcurve(solution_norm,residual_norm)

# def plot_train_test_errors(train_errors, test_errors, lambd_values):
#     plt.plot(lambd_values, train_errors, label="Train error")
#     plt.plot(lambd_values, test_errors, label="Test error")
#     plt.xscale("log")
#     plt.legend(loc="upper left")
#     plt.xlabel(r"$\lambda$", fontsize=16)
#     plt.title("Mean Squared Error (MSE)")
#     plt.show()

# # plot_train_test_errors(train_errors, test_errors, lambd_values)

# def plot_regularization_path(lambd_values, x_values):
#     num_coeffs = len(x_values[0])
#     for i in range(num_coeffs):
#         plt.plot(lambd_values, [wi[i] for wi in x_values])
#     plt.xlabel(r"$\lambda$", fontsize=16)
#     plt.xscale("log")
#     plt.title("Regularization Path")
#     plt.show()

# plot_regularization_path(lambd_values, x_values)

#%% Ridge regression with cvxpy and matlab data
# def plot_sol(x_exact, x_estimated):
#     plt.plot(x_exact, label= 'exact')
#     for x_e in x_estimated:
#         plt.plot(x_e, label= 'estimated')
#     plt.legend(loc="upper left")
#     plt.show()

# def plot_lcurve(solution_norm, residual_norm):
#     # plt.plot(np.log(residual_norm), np.log(solution_norm))
#     plt.plot(residual_norm, solution_norm, 'o')
#     plt.xscale("log")
#     plt.yscale("log")
#     # plt.legend(loc="upper left")
#     plt.xlabel('|Hx-p|', fontsize=16)
#     plt.ylabel("|x|")
#     plt.show()

# def lcurve_der(lambd_values, solution_norm, residual_norm):
#     dxi = (np.array(solution_norm[1:])**2 - np.array(solution_norm[0:-1])**2)/\
#         (np.array(lambd_values[1:]) - np.array(lambd_values[0:-1]))
#     dpho = (np.array(residual_norm[1:])**2 - np.array(residual_norm[0:-1])**2)/\
#         (np.array(lambd_values[1:]) - np.array(lambd_values[0:-1]))
#     clam = (2**np.array(lambd_values[1:])*(dxi**2))/\
#         ((dpho**2 + dxi**2)**3/2)
#     id_maxcurve = np.where(clam == np.amax(clam))
#     print('The ideal value of lambda is: {}'.format(lambd_values[id_maxcurve[0]+1]))
#     plt.plot(lambd_values[1:], clam)
#     plt.show()

# import scipy.io
# mat = scipy.io.loadmat('data/test_tik.mat')
# H = np.array(mat['A'])
# x_exact = np.reshape(np.array(mat['x']), 64)
# print('norm 2 of x: {}'.format(np.linalg.norm(x_exact, ord=2)))
# pm = np.reshape(np.array(mat['b']), 64) + 0.005*np.random.randn(64)

# x = cp.Variable(H.shape[1])
# lambd = cp.Parameter(nonneg=True)
# problem = cp.Problem(cp.Minimize(objective_fn(H, pm, x, lambd)))
# lambd_values = np.logspace(-5, 1, 100)

# x_values = []
# solution_norm = []
# residual_norm = []
# for v in lambd_values:
#     lambd.value = v
#     problem.solve()
#     x_values.append(x.value)
#     # plot_sol(pm, x.value)
#     solution_norm.append(np.linalg.norm(x.value, ord=2)) #+= [(lm.coef_**2).sum()]
#     residual_norm.append(np.linalg.norm(H @ x.value - pm, ord=2)) #+= [((lm.predict(H) - pm)**2).sum()]

# ind =35
# print('norm 2 of x pred: {}'.format(np.linalg.norm(x_values[ind], ord=2)))# np.sum((np.abs(x_values[0]))**2)))
# # plot_sol(pm, x_values)
# plot_lcurve(solution_norm, residual_norm)
# # plt.show()

# lcurve_der(lambd_values, solution_norm, residual_norm)

#%% Ridge regression with sklearn
# from sklearn.linear_model import Ridge
# # Data
# m = 20
# n = 15
# np.random.seed(1)
# H = np.random.randn(m, n) #+ 1j*np.random.randn(m, n)
# pm = np.random.randn(m) #+ 1j*np.random.randn(m)

# lambd = np.logspace(-10, 10, 1000)
# solution_norm = []
# residual_norm = []

# for lam in lambd: 
#     lm = Ridge(alpha=lam)
#     lm.fit(H, pm)
#     solution_norm += [(lm.coef_**2).sum()]
#     residual_norm += [((lm.predict(H) - pm)**2).sum()]

# plt.loglog(residual_norm, solution_norm, 'k-')
# plt.show()

#%%
# Generate a random non-trivial linear program.
# m = 15
# n = 10
# np.random.seed(1)
# s0 = np.random.randn(m)
# print('Initial s0 is: {}'.format(s0))
# lamb0 = np.maximum(-s0, 0)
# print('lamb0 is: {}'.format(lamb0))
# s0 = np.maximum(s0, 0)
# print('new s0 is: {}'.format(s0))
# x0 = np.random.randn(n)
# A = np.random.randn(m, n)
# b = A@x0 + s0
# c = -A.T@lamb0

# # Define and solve the CVXPY problem.
# x = cp.Variable(n)
# prob = cp.Problem(cp.Minimize(c.T@x), [A@x <= b])
# prob.solve()

# # Print result.
# print("\nThe optimal value is", prob.value)
# print("A solution x is")
# print(x.value)
# print("A dual solution is")
# print(prob.constraints[0].dual_value)














# # Problem data.
# n = 15
# m = 10
# np.random.seed(1)
# A = np.random.randn(n, m)
# b = np.random.randn(n)
# print('shape of A: {}'.format(A.shape))
# print('shape of B: {}'.format(b.shape))
# # gamma must be nonnegative due to DCP rules.
# gamma = cp.Parameter(nonneg=True)

# # Construct the problem.
# x = cp.Variable(m)
# error = cp.sum_squares(A*x - b)
# obj = cp.Minimize(error + gamma*cp.norm(x, 1))
# prob = cp.Problem(obj)

# # Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
# sq_penalty = []
# l1_penalty = []
# x_values = []
# gamma_vals = np.logspace(-4, 6)
# for val in gamma_vals:
#     gamma.value = val
#     prob.solve()
#     # Use expr.value to get the numerical value of
#     # an expression in the problem.
#     sq_penalty.append(error.value)
#     l1_penalty.append(cp.norm(x, 1).value)
#     x_values.append(x.value)

# plt.plot(sq_penalty, l1_penalty)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.ylabel('||x||_1', fontsize=16)
# plt.xlabel('|Ax-b|^2', fontsize=16)
# plt.show()

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.figure(figsize=(6,10))

# # Plot trade-off curve.
# # plt.subplot(211)
# plt.plot(l1_penalty, sq_penalty)
# # plt.xlabel('||x||_1', fontsize=16)
# # plt.ylabel('|Ax-b|^2', fontsize=16)
# # plt.title('Trade-Off Curve for LASSO', fontsize=16)

# # Plot entries of x vs. gamma.
# # plt.subplot(212)
# # for i in range(m):
# #     plt.plot(gamma_vals, [xi[i] for xi in x_values])
# # plt.xlabel('gamma', fontsize=16)
# # plt.ylabel('x_{i}', fontsize=16)
# # plt.xscale('log')
# # plt.title('text{Entries of x vs. }gamma', fontsize=16)

# # plt.tight_layout()
# plt.show()

#%%
# # Problem data.
# m = 10
# n = 5
# np.random.seed(1)
# A = np.random.randn(m, n)
# b = np.random.randn(m)
# print('shape of A: {}'.format(A.shape))
# print('shape of B: {}'.format(b.shape))
# # Construct the problem.
# x = cp.Variable(n)
# objective = cp.Minimize(cp.sum_squares(A*x - b))
# constraints = [0 <= x, x <= 1]
# prob = cp.Problem(objective, constraints)

# print("Optimal value", prob.solve())
# print("Optimal var")
# print(x.value) # A numpy ndarray.


# # Create two scalar optimization variables.
# x = cp.Variable()
# y = cp.Variable()

# # Create two constraints. logical list
# constraints = [x + y == 1,
#     x - y >= 1]

# # Form objective.
# obj = cp.Minimize((x - y)**2)

# # Form and solve problem.
# prob = cp.Problem(obj, constraints)
# prob.solve()  # Returns the optimal value.
# print("status:", prob.status)
# print("optimal value", prob.value)
# print("optimal var", x.value, y.value)

# prob2 = cp.Problem(cp.Maximize(x + y), prob.constraints)
# print("optimal value", prob2.solve())

# # Replace the constraint (x + y == 1).
# constraints = [x + y <= 3] + prob2.constraints[1:]
# prob3 = cp.Problem(prob2.objective, constraints)
# print("optimal value", prob3.solve())
# # print('prob3 constraints: {}'.format(prob3.constraints))

# x = cp.Variable()

# # An infeasible problem.
# prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= 0])
# prob.solve()
# print("status:", prob.status)
# print("optimal value", prob.value)

# # An unbounded problem.
# prob = cp.Problem(cp.Minimize(x))
# prob.solve()
# print("status:", prob.status)
# print("optimal value", prob.value)