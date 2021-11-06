# -*- coding: utf-8 -*-
from pyDACP import core, chebyshev
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, eig
from scipy.sparse import eye, diags

# +
N = 1000

a = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5)
b = 2 * (np.random.rand(N) - 0.5)

H = diags(a, offsets=-1) + diags(b, offsets=0) + diags(a.conj(), offsets=1)

plt.matshow(H.real.toarray())
# -

# %%time
dacp=core.DACP_reduction(H, a=0.5, eps=0.05, bounds=None, sampling_subspace=3)

true_eigvals, true_eigvecs = eig(H.todense())
ψ_proj = dacp.get_filtered_vector()

plt.scatter(np.real(true_eigvals), np.log(np.abs(true_eigvecs.T.conj()@ψ_proj)))
plt.xlim(-2*dacp.a, 2*dacp.a)

# %%time
ham_red=dacp.get_subspace_matrix()

plt.matshow(ham_red.real)

# +
import kwant

dos = kwant.kpm.SpectralDensity(
    H,
    mean=True,
    energy_resolution=0.005
)

dos_red = kwant.kpm.SpectralDensity(
    ham_red,
    mean=True,
    energy_resolution=0.005
)
# -

plt.plot(*dos())
plt.plot(*dos_red())
plt.axvline(-dacp.a, c='k')
plt.axvline(dacp.a, c='k')
plt.xlim(-2*dacp.a, 2*dacp.a)
plt.axhline(0, c='k', ls='--')
plt.show()
