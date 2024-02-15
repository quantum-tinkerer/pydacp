import numpy as np
from dacp.dacp import eigvalsh, estimated_errors
from scipy.sparse import csr_matrix, kron
from scipy.sparse import diags, eye
import pytest

# +
N = 1000
N_block = 200
deg_n = 5
loop_n = 2
window_size = 0.1
window=[-window_size, window_size]
k=12
tol=1e-4


def random_ham(N):
    c = 2 * (np.random.rand(N - 1) + np.random.rand(N - 1) * 1j - 0.5 * (1 + 1j))
    b = 2 * (np.random.rand(N) - 0.5)

    H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)
    return csr_matrix(H)

def random_ham_deg(N, deg):
    H = random_ham(N)
    return csr_matrix(kron(H, eye(deg)))

def eigv_errors(H, window, **dacp_kwargs):
    eigv, _ = np.linalg.eigh(H.todense())
    if dacp_kwargs:
        eigv_dacp = eigvalsh(H, window, **dacp_kwargs)
    else:
        eigv_dacp = eigvalsh(H, window)

    map_eigv = []
    for value in eigv_dacp:
        closest = np.abs(eigv - value).min()
        map_eigv.append(eigv[np.abs(eigv - value) == closest][0])
    map_eigv = np.array(map_eigv)

    map_eigv = map_eigv[np.argsort(np.abs(map_eigv))]
    eigv_dacp = eigv_dacp[np.argsort(np.abs(eigv_dacp))]

    relative_error = np.abs((eigv_dacp - map_eigv) / map_eigv)[map_eigv < window_size]
    eta = estimated_errors(eigvals=eigv_dacp, tol=tol, filter_order=k, window=window)

    diff = relative_error - eta
    return np.log10(np.heaviside(diff, 0) * diff / eta)


def eigv_errors_test(deg=False, **dacp_kwargs):
    relative_error_list = []
    if deg:
        H = random_ham_deg(N_block, deg_n)
    else:
        H = random_ham(N)

    relative_error = eigv_errors(H, window, **dacp_kwargs)
    if relative_error.size != 0:
        relative_error_list.append(np.max(relative_error))

    return np.asarray(relative_error_list)

@pytest.mark.repeat(loop_n)
def test_eigvals():
    """
    Test the eigenvalue with non-degenerate hamiltonians
    """
    error_diff = eigv_errors_test()
    assert error_diff.any(), "Errors don't match the theoretical value."

@pytest.mark.repeat(loop_n)
def test_eigvals_deg():
    """
    Test the eigenvalue method with degenerate hamiltonians
    """
    error_diff = eigv_errors_test(deg=True, random_vectors=2)
    assert error_diff.any(), "Errors don't match the theoretical value."