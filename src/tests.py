import numpy as np
import matplotlib.pyplot as plt
from pyDACP.core import dacp_eig
from scipy.sparse import csr_matrix

# +
N = 200
window_size = 0.1

def random_values(shape):
    # defines random values from -1 to 1
    return (np.random.rand(*shape)-1/2)*2

H = random_values((N,N)) + random_values((N,N))*1j
H = (H.conj().T + H)/(2*np.sqrt(N))

H = csr_matrix(H)


# +
def eigv_errors(H, window_size, **dacp_kwargs):
    eigv, eigs = np.linalg.eigh(H.todense())
    if dacp_kwargs:
        eigv_dacp = dacp_eig(H, window_size, **dacp_kwargs)
    else:
        eigv_dacp = dacp_eig(H, window_size)

    N_dacp = len(eigv_dacp)

    eigv = eigv[np.abs(eigv) < window_size]
    eigv = eigv[np.argsort(np.abs(eigv))]
    eigv_dacp = eigv_dacp[np.argsort(np.abs(eigv_dacp))]

    relative_error = np.abs(eigv_dacp-eigv[:N_dacp])/np.abs(eigv[:N_dacp])
    missed_evals = eigv[N_dacp:]

    return relative_error, missed_evals


def eigs_errors(H, window_size, **dacp_kwargs):
    eigv, eigs = np.linalg.eigh(H.todense())
    if dacp_kwargs:
        eigv_dacp, eigs_dacp = dacp_eig(H, window_size, return_eigenvectors=True, **dacp_kwargs)
    else:
        eigv_dacp, eigs_dacp = dacp_eig(H, window_size, return_eigenvectors=True)

    N_dacp = len(eigv_dacp)

    window_args = np.abs(eigv) < window_size
    eigv = eigv[window_args]
    eigs = eigs[window_args, :]

    order_args = np.argsort(np.abs(eigv))
    eigv = eigv[order_args]
    eigs = eigs[order_args, :]

    order_args = np.argsort(np.abs(eigv_dacp))
    eigv_dacp = eigv_dacp[order_args]
    eigs_dacp = eigs_dacp[order_args, :]

    relative_error = np.abs(eigv_dacp-eigv[:N_dacp])/np.abs(eigv[:N_dacp])
    missed_evals = eigv[N_dacp:]

    r = np.einsum('ij, kj -> ki', H.todense(), eigs_dacp) - np.einsum('i, ij -> ij', eigv_dacp, eigs_dacp)

    return relative_error, missed_evals, r


# -

error, evals, r = eigs_errors(H, window_size)

np.max(np.linalg.norm(r, axis=1))

plt.plot(np.log10(error))


