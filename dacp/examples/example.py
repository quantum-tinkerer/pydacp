# -*- coding: utf-8 -*-
# %%
from dacp.dacp import estimated_errors
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse import kron

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["lines.linewidth"] = 0.65
plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18

# %%
N = int(5e2)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
H = diags(b, offsets=0)

# %%
# %%time
true_vals, true_vecs=np.linalg.eigh(H.todense())

Emax = np.max(true_vals)
true_vals /= Emax
H /= Emax

# %%
# %%time
k=12
a = 0.1
tol=1e-3
window=[-a, 2*a]
evals = eigvalsh(
    H,
    window=window,
    random_vectors=10,
    filter_order=k,
    tol=tol
)

# %%
print(np.min(evals), np.max(evals))

# %%
map_eigv=[]
for value in evals:
    closest = np.abs(true_vals-value).min()
    map_eigv.append(true_vals[np.abs(true_vals-value) == closest][0])
true_vals = np.array(map_eigv)

# %%
true_vals=np.sort(true_vals)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, true_vals, c='r', s=4)
plt.ylabel(r'$E_i$')
plt.xlabel(r'$n$')
plt.show()

# %%
error = np.abs((true_vals - evals)/evals)
true_vals=np.sort(true_vals)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c=np.log10(error), s=20, cmap='inferno')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.colorbar()
plt.scatter(n_true, true_vals, c='k', s=2)
plt.ylabel(r'$E_i$')
plt.xlabel(r'$n$')
plt.show()

# %%
plt.scatter(evals, np.abs(true_vals - evals))
plt.ylabel(r'$\delta E_i$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.axhline(np.finfo(float).eps, ls='--', c='k')
plt.axhline(1/N, ls='--', c='k')
plt.show()

# %%
window_size = (window[1] - window[0]) / 2
sigma = (window[1] + window[0]) / 2
delta = np.finfo(float).eps
alpha = 1 / (4 * k) * np.log(tol * window_size / np.finfo(float).eps)
a_w = window_size / np.sqrt(2 * alpha - alpha**2)
Ei = np.linspace(window[0], window[1], 300)
c_i_sq = np.exp(4 * k * np.sqrt(a_w**2 - (Ei - sigma)**2) / a_w)
eta = delta * np.exp(4 * k) / (np.abs(Ei) * c_i_sq)

# %%
plt.plot(Ei, eta, 'r')
plt.fill_between(Ei, 0.01*eta, 100*eta, alpha=0.4, fc='r')
plt.scatter(evals, np.abs((true_vals - evals)/evals), c='k', zorder=10, s=1)
plt.ylabel(r'$|\delta E_i/E_i|$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.xlim(window[0], window[1])
plt.show()

# %%
np.max(true_vals)

# %%
plt.scatter(evals, estimated_errors(evals, window), c='b', s=10, marker='+')
plt.plot(Ei, eta, 'r')
plt.ylabel(r'$|\delta E_i/E_i|$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.xlim(window[0], window[1])
plt.show()

# %% [markdown]
# ## Degenerate case

# %%
N = int(250)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)

# %%
H = kron(H, eye(4))

# %%
# %%time
window=[-a, a]
evals = eigvalsh(
    H,
    window=window,
    random_vectors=5,
    filter_order=k,
    tol=tol
)

# %%
# %%time
true_vals=np.linalg.eigvalsh(H.todense())

# %%
map_eigv=[]
for value in evals:
    closest = np.abs(true_vals-value).min()
    map_eigv.append(true_vals[np.abs(true_vals-value) == closest][0])
true_vals = np.array(map_eigv)

# %%
true_vals=np.sort(true_vals)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, true_vals, c='r', s=4)
plt.ylabel(r'$E_i$')
plt.xlabel(r'$n$')
plt.show()

# %%
plt.scatter(evals, np.abs(true_vals - evals))
plt.ylabel(r'$\delta E_i$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.axhline(np.finfo(float).eps, ls='--', c='k')
plt.show()

# %%
window_size = (window[1] - window[0]) / 2
sigma = (window[1] + window[0]) / 2
delta = np.finfo(float).eps
alpha = 1 / (4 * k) * np.log(tol * window_size / np.finfo(float).eps)
a_w = window_size / np.sqrt(2 * alpha - alpha**2)
Ei = np.linspace(window[0], window[1], 300)
c_i_sq = np.exp(4 * k * np.sqrt(a_w**2 - (Ei - sigma)**2) / a_w)
eta = delta * np.exp(4 * k) / (np.abs(Ei) * c_i_sq)

# %%
plt.plot(Ei, eta, 'r')
plt.fill_between(Ei, 0.01*eta, 100*eta, alpha=0.4, fc='r')
plt.scatter(evals, np.abs((true_vals - evals)/evals), c='k', zorder=10, s=1)
plt.ylabel(r'$|\delta E_i/E_i|$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.xlim(-a, a)
plt.show()
