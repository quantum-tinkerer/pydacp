from pyDACP import core
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import eye, diags
import math
from scipy.linalg.lapack import zlarf, zlarfg

# +
N = 500
a = 0.2
c = np.random.rand(N-1) + np.random.rand(N-1)*1j
b = np.random.rand(N)
H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)

dacp=core.DACP_reduction(H, a=a, eps=0.1, bounds=None, sampling_subspace=5)
# -

dacp.estimate_subspace_dimenstion()


def basis(v_proj, matrix, indices):
    v_basis = []
    count = 0
    N = matrix.shape[0]
    P = np.eye(N)
    Pi = np.eye(N)
    # TODO: If k is too large, the norms of the vectors are from some large order.
    k = indices[-1]
    for i in range(k+1):
        if i == 0:
            v_n = v_proj
        elif i == 1:
            v_nm1 = v_n
            v_n = matrix @ v_nm1
        else:
            v_np1 = 2 * matrix @ v_n - v_nm1
            v_nm1 = v_n
            v_n = v_np1
        if i in indices:
            v_n_hh = v_n/np.linalg.norm(v_n)
            r = np.linalg.norm((P @ v_n_hh)[count+1:])
            if np.isclose(r, 0):
                print('Ended with ' + str(count) + ' vectors.')
                basis = (P@Pi).T.conj()[:, :count]
                return basis/(np.linalg.norm(basis, axis=0))
            else:
                v_n_hh = P @ v_n_hh
                Pi = P @ Pi
                beta, v_orth, tau = zlarfg(N-count, v_n_hh[count], v_n_hh[count+1:])
                v_orth = np.array([1, *v_orth])
                P = (np.eye(N - count) - tau * np.outer(v_orth, v_orth.conj())) / beta
                P = scipy.linalg.block_diag(np.eye(count), P)
                count += 1
    basis=(P@Pi).T.conj()[:, :count]
    return basis/(np.linalg.norm(basis, axis=0))


v_proj=dacp.get_filtered_vector()
d = dacp.estimate_subspace_dimenstion()
n = math.ceil(np.abs((d*dacp.sampling_subspace - 1)/2))
a_r = dacp.a / np.max(np.abs(dacp.bounds))
n_array = np.arange(1, n+1, 1)
indicesp1 = (n_array*np.pi/a_r).astype(int)
indices = np.sort(np.array([*indicesp1, *indicesp1-1]))
vbasis=basis(v_proj=v_proj, matrix=dacp.G_operator(), indices=indices)

np.shape(vbasis)

indices.shape[0]

plt.matshow(np.abs(vbasis.conj().T@vbasis))
plt.colorbar()
