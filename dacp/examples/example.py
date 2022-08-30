# -*- coding: utf-8 -*-
# +
from dacp.dacp import eigh
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
from scipy.sparse import block_diag, kron

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["lines.linewidth"] = 0.65
plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18

# +
N = int(5e2)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
# -

# %%time
true_vals, true_vecs=np.linalg.eigh(H.todense())
Emax = np.max(true_vals)
true_vals /= Emax
H /= Emax

# %%time
k=12
a = 0.1
tol=1e-3
error_window = 0.1
evals = eigh(
    H,
    window=[-a,a],
    random_vectors=10,
    return_eigenvectors=False,
    filter_order=k,
    tol=tol
)

map_eigv=[]
for value in evals:
    closest = np.abs(true_vals-value).min()
    map_eigv.append(true_vals[np.abs(true_vals-value) == closest][0])
true_vals = np.array(map_eigv)

true_vals=np.sort(true_vals)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, true_vals, c='r', s=4)
plt.ylabel(r'$E_i$')
plt.xlabel(r'$n$')
plt.show()

error = np.abs((true_vals - evals)/evals)
true_vals=np.sort(true_vals)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c=np.log10(error), s=20, cmap='RdBu_r')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.colorbar()
plt.scatter(n_true, true_vals, c='k', s=2)
plt.ylabel(r'$E_i$')
plt.xlabel(r'$n$')
plt.show()

plt.scatter(evals, np.abs(true_vals - evals))
plt.ylabel(r'$\delta E_i$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.axhline(np.finfo(float).eps, ls='--', c='k')
plt.axhline(1/N, ls='--', c='k')
plt.show()

delta = np.finfo(float).eps
alpha = 1 / (4 * k) * np.log(tol * a / np.finfo(float).eps)
a_w = a / np.sqrt(2 * alpha - alpha**2)
Ei = np.linspace(-a_w, a_w, 300)
c_i_sq = np.exp(4 * k * np.sqrt(a_w**2 - Ei**2) / a_w)
eta = delta * np.exp(4 * k) / (np.abs(Ei) * c_i_sq)

plt.plot(Ei, eta, 'r')
plt.fill_between(Ei, 0.01*eta, 100*eta, alpha=0.4, fc='r')
plt.scatter(evals, np.abs((true_vals - evals)/evals), c='k', zorder=10, s=1)
plt.ylabel(r'$|\delta E_i/E_i|$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.xlim(-a, a)
plt.show()

# ## Degenerate case

# +
N = int(250)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
# -

H = kron(H, eye(4))

# %%time
evals = eigh(
    H,
    window_size=a,
    eps=0.05,
    random_vectors=5,
    return_eigenvectors=False,
    filter_order=k,
    error_window=error_window
)

# %%time
true_vals=np.linalg.eigvalsh(H.todense())

map_eigv=[]
for value in evals:
    closest = np.abs(true_vals-value).min()
    map_eigv.append(true_vals[np.abs(true_vals-value) == closest][0])
true_vals = np.array(map_eigv)

true_vals=np.sort(true_vals)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, true_vals, c='r', s=4)
plt.ylabel(r'$E_i$')
plt.xlabel(r'$n$')
plt.show()

plt.scatter(evals, np.abs(true_vals - evals))
plt.ylabel(r'$\delta E_i$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.axhline(np.finfo(float).eps, ls='--', c='k')
plt.show()

delta = np.finfo(float).eps
a_w = a * (1 + error_window)
Ei = np.linspace(-a_w, a_w, 300)
c_i_sq = np.exp(4 * k * np.sqrt(a_w**2 - Ei**2) / a_w)
eta = delta * np.exp(4 * k) / (np.abs(Ei) * c_i_sq)

plt.plot(Ei, eta, 'r')
plt.fill_between(Ei, 0.01*eta, 100*eta, alpha=0.4, fc='r')
plt.scatter(evals, np.abs((true_vals - evals)/evals), c='k', zorder=10, s=1)
plt.ylabel(r'$|\delta E_i/E_i|$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.xlim(-a, a)
plt.show()
