# -*- coding: utf-8 -*-
# +
from dacp.dacp import eigh
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
from scipy.sparse import block_diag

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["lines.linewidth"] = 0.65
plt.rcParams["font.size"] = 16
plt.rcParams["legend.fontsize"] = 16

# +
N = int(1e3)
np.random.seed(1)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
# -

# %%time
evals, _ = eigh(
    H, window_size=0.1, eps=0.05, random_vectors=2, return_eigenvectors=True, filter_order=14
)

# %%time
true_vals, true_vecs=eigsh(H, return_eigenvectors=True, sigma=0, k=evals.shape[0])

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
# plt.xlim(-0.1, 0.1)
# plt.yscale('log')
plt.show()

# ## Degenerate case

a = diags(np.linspace(1, 3))
from scipy.sparse import block_diag
H = block_diag((a, -a))

plt.matshow(H.todense())

# %%time
evals, _ = eigh(
    H, window_size=0.1, eps=0.05, random_vectors=2, return_eigenvectors=True, filter_order=14
)
