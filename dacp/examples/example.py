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
N = int(2e3)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
# -

# %%time
true_vals=np.linalg.eigvalsh(H.todense())
Emax = np.max(true_vals)
true_vals /= Emax
H /= Emax

# %%time
k=16
a = 0.1
evals = eigh(
    H,
    window_size=a,
    eps=0.05,
    random_vectors=10,
    return_eigenvectors=False,
    filter_order=k,
    error_window=0.,
)

true_vals = np.sort(true_vals[np.argsort(np.abs(true_vals))][:len(evals)])

print(len(evals), len(true_vals))

true_vals=np.sort(true_vals)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, true_vals, c='r', s=4)
# plt.ylim(-0.1, 0.1)
# plt.xlim(np.min(n), np.max(n))
plt.ylabel(r'$E_i$')
plt.xlabel(r'$n$')
plt.show()

plt.scatter(evals, np.abs(true_vals - evals))
plt.ylabel(r'$\delta E_i$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.show()

delta = (np.sqrt(N) * np.exp(-2 * k))**(1.4)

# delta = 1.2e-16
Ei = np.linspace(-a, a, 300)
c_i_sq = np.exp(4*k*np.sqrt(a**2-Ei**2)/a)
eta = delta*np.exp(4*k)/(np.abs(Ei)*c_i_sq)

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
N = int(1e2)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
# -

H = kron(H, eye(4))

# %%time
evals = eigh(
    H,
    window_size=0.1,
    eps=0.05,
    random_vectors=2,
    return_eigenvectors=False,
    filter_order=14,
    error_window=0.25,
)

# %%time
true_vals=np.linalg.eigvalsh(H.todense())
true_vals = true_vals[np.abs(true_vals) < 0.1]

print(len(true_vals), len(evals))
len(true_vals) == len(evals)

true_vals=np.sort(true_vals)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, true_vals, c='r', s=4)
# plt.ylim(-0.1, 0.1)
# plt.xlim(np.min(n), np.max(n))
plt.ylabel(r'$E_i$')
plt.xlabel(r'$n$')
plt.show()

plt.scatter(evals, np.abs(true_vals - evals))
plt.ylabel(r'$\delta E_i$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.show()

plt.scatter(evals, np.abs((true_vals - evals)/evals))
plt.ylabel(r'$\delta E_i$')
plt.xlabel(r'$E_i$')
plt.yscale('log')
plt.show()
