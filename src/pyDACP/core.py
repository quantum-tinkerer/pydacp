from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, csr_matrix
from scipy.linalg import eigh, eigvalsh, qr, qr_insert
from . import chebyshev
import numpy as np
from math import ceil
import matplotlib.pyplot as plt


def svd_decomposition(S, matrix_proj):
    s, V = eigh(S)
    indx = s > 1e-12
    lambda_s = np.diag(1 / np.sqrt(s[indx]))
    U = V[:, indx] @ lambda_s
    return U.T.conj() @ matrix_proj @ U


def dacp_eig(
    matrix,
    a,
    eps=0.1,
    bounds=None,
    random_vectors=2,
    return_eigenvectors=False,
    filter_order=12,
):
    """
    Find the eigendecomposition within the given spectral bounds of a given matrix.

    Parameters
    ----------
    matrix : 2D array or sparse matrix
        Initial matrix.
    eps : float
        Ensures that the bounds are strict.
    bounds : tuple, or None
        Boundaries of the spectrum. If not provided the maximum and
        minimum eigenvalues are calculated.
    random_vectors : int
        When return_eigenvectors=False, specifies the maximum expected
        degeneracy of the matrix.
    return_eigenvectors : bool
        If True, returns eigenvectors and processes general degeneracies.
        However, if False, the algorithm conserves memory
        and only processes random_vectors>degenerecies.
    filter_order : int
        The number of times a vector is filtered is given by filter_order*E_max/a.
    """
    matrix = csr_matrix(matrix)

    if bounds is None:
        # Relative tolerance to which to calculate eigenvalues.  Because after
        # rescaling we will add eps / 2 to the spectral bounds, we don't need
        # to know the bounds more accurately than eps / 2.
        tol = eps / 2

        lmax = float(eigsh(matrix, k=1, which="LA", return_eigenvectors=False, tol=tol))
        lmin = float(eigsh(matrix, k=1, which="SA", return_eigenvectors=False, tol=tol))

        if lmax - lmin <= abs(lmax + lmin) * tol / 2:
            raise ValueError(
                "The matrix has a single eigenvalue, it is not possible to "
                "obtain a spectral density."
            )

        bounds = [lmin, lmax]

    Emin = bounds[0] * (1 + eps)
    Emax = bounds[1] * (1 + eps)
    E0 = (Emax - Emin) / 2
    Ec = (Emax + Emin) / 2
    G_operator = (matrix - eye(matrix.shape[0]) * Ec) / E0

    Emax = np.max(np.abs(bounds)) * (1 + eps)
    E0 = (Emax ** 2 - a ** 2) / 2
    Ec = (Emax ** 2 + a ** 2) / 2
    F_operator = (matrix @ matrix - eye(matrix.shape[0]) * Ec) / E0

    def get_filtered_vector():
        v_rand = 2 * (
            np.random.rand(matrix.shape[0], random_vectors)
            + np.random.rand(matrix.shape[0], random_vectors) * 1j
            - 0.5 * (1 + 1j)
        )
        v_rand = v_rand / np.linalg.norm(v_rand, axis=0)
        K_max = int(filter_order * np.max(np.abs(bounds)) / a)
        vec = chebyshev.low_E_filter(v_rand, F_operator, K_max)
        return vec / np.linalg.norm(vec, axis=0)

    a_r = a / np.max(np.abs(bounds))
    dk = ceil(np.pi / a_r)
    a_r = a / np.max(np.abs(bounds))

    if return_eigenvectors:
        # First run
        Q, R = chebyshev.basis(
            v_proj=get_filtered_vector(), G_operator=G_operator, dk=dk
        )
        # Second run
        Qi, Ri = chebyshev.basis(
            v_proj=get_filtered_vector(),
            G_operator=G_operator,
            dk=dk,
            Q=Q,
            R=R,
            first_run=False,
        )
        # Other runs to solve higher degeneracies
        while Q.shape[1] < Qi.shape[1]:
            Q, R = Qi, Ri
            Qi, Ri = chebyshev.basis(
                v_proj=get_filtered_vector(),
                G_operator=G_operator,
                dk=dk,
                Q=Q,
                R=R,
                first_run=False,
            )
        v_basis = Q
        matrix_proj = v_basis.conj().T @ matrix.dot(v_basis)
        eigvals, eigvecs = eigh(matrix_proj)
        return eigvals, eigvecs @ v_basis.T

    else:
        N_loop = 0
        n_evolution = False
        while True:
            v_proj = get_filtered_vector()
            if N_loop == 0:
                v_0, k_list, S, matrix_proj = chebyshev.eigvals_init(
                    v_proj, G_operator, matrix, dk
                )
                N_H_prev = sum(np.diag(np.invert(np.isclose(qr(S, mode='r')[0], 0))))
                new_vals = N_H_prev
            else:
                if new_vals <= random_vectors and not n_evolution:
                    n_evolution = N_loop
                v_0, S, matrix_proj = chebyshev.eigvals_deg(
                    v_0, v_proj, k_list, S, matrix_proj, G_operator, matrix, dk, n_evolution
                )
                N_H_cur = sum(np.diag(np.invert(np.isclose(qr(S, mode='r')[0], 0))))
                new_vals = N_H_cur - N_H_prev
                if new_vals > 0:
                    N_H_prev = N_H_cur
                else:
                    H_red = svd_decomposition(S, matrix_proj)
                    return eigvalsh(H_red)
            N_loop += 1
