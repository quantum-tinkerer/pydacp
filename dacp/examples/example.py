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
plt.rcParams["font.size"] = 16
plt.rcParams["legend.fontsize"] = 16

# +
N = int(4e2)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
# -

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
