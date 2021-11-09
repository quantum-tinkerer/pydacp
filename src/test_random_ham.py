# -*- coding: utf-8 -*-
from pyDACP import core, chebyshev
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.linalg import eigh, eig
from scipy.sparse import eye, diags

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["lines.linewidth"] = 0.65
plt.rcParams["font.size"] = 16
plt.rcParams["legend.fontsize"] = 16

# +
N = 1000

c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)

plt.matshow(H.real.toarray())
plt.show()
# -

# %%time
dacp=core.DACP_reduction(H, a=0.2, eps=0.05, bounds=None, sampling_subspace=1.5)

true_eigvals, true_eigvecs = eig(H.todense())
v_proj = dacp.get_filtered_vector()

plt.scatter(np.real(true_eigvals), np.log(np.abs(true_eigvecs.T.conj()@v_proj)), c='k')
plt.xlim(-3*dacp.a, 3*dacp.a)
plt.axvline(-dacp.a, ls='--', c='k')
plt.axvline(dacp.a, ls='--', c='k')
plt.ylabel(r'$|c|$')
plt.xlabel(r'$E$')
plt.show()

# %%time
ham_red=dacp.get_subspace_matrix()

plt.matshow(ham_red.real)
plt.show()

red_eigvals, red_eigvecs = eig(ham_red)

# This part is quite unstable.
# Half of the bandwidth
half_bandwidth=np.max(np.abs(true_eigvals))
# Number 50 times smaller than level spacing.
err=half_bandwidth/N/2
# Get indices
indx1=np.min(red_eigvals)-err<=true_eigvals
indx2=true_eigvals<=np.max(red_eigvals)+err
indx=indx1 * indx2
# Extract real eigenvalues within the desired window.
window_eigvals=true_eigvals[indx]

plt.plot(np.sort(window_eigvals), np.sort(red_eigvals), '-o', c='k')
plt.xlabel(r'$E_n^{big}$')
plt.ylabel(r'$E_n^{small}$')
plt.xlim(-dacp.a, dacp.a)
plt.ylim(-dacp.a, dacp.a)
plt.show()

# res=np.sort(red_eigvals).copy()[:window_eigvals.shape[0]]
# res-=np.sort(window_eigvals)
res=np.sort(red_eigvals)-np.sort(window_eigvals)
plt.plot(np.sort(window_eigvals), np.log(res/dacp.a), '-o', c='k')
plt.ylabel(r'$\log(E_n^{big} - E_n^{small})$')
plt.xlabel(r'$E_n^{big}$')
plt.xlim(-dacp.a, dacp.a)
plt.show()
