from scipy.sparse.linalg import eigsh
from scipy.sparse import eye, csr_matrix
import scipy.linalg
import numpy as np
from math import ceil
import itertools as it
import warnings

import matplotlib.pyplot as plt


def svd_decomposition(S, matrix_proj):
    """
    Perform SVD decomposition.

    Parameters:
    -----------

    S : ndarray
        Overlap matrix.
    matrix_proj : ndarray
        Projected matrix.
    """
    s, V = scipy.linalg.eigh(S)
    indx = s > 1e-12
    lambda_s = np.diag(1 / np.sqrt(s[indx]))
    U = V[:, indx] @ lambda_s
    return U.T.conj() @ matrix_proj @ U


def chebyshev_recursion_gen(matrix, v_0):
    """
    Recursively apply Chebyshev polynomials of a matrix.

    Parameters
    ----------
    matrix : sparse matrix
        Compute Chebyshev polynomials of this matrix.
    v_0 : 1D array
        Initial vector.
    """
    order = 0
    while True:
        if order == 0:
            v_n = v_0
        elif order == 1:
            v_nm1 = v_n
            v_n = matrix.dot(v_nm1)
        else:
            v_np1 = 2 * matrix.dot(v_n) - v_nm1
            v_nm1 = v_n
            v_n = v_np1
        order += 1
        yield v_n


def low_E_filter(v_rand, F_operator, K_max):
    """
    Chebyshev filter of a radom vector `v_proj`.

    Parameters
    ----------
    vproj : 2D array
        Collection of random vectors.
    F_operator : sparse matrix
        Filter operator.
    K_max : int
        Highest order of Chebyshev polynomials of `F_operator` computed.
    """
    chebyshev_recursion = chebyshev_recursion_gen(F_operator, v_rand)
    for i in range(K_max + 1):
        v_n = next(chebyshev_recursion)
    return v_n / np.linalg.norm(v_n, axis=0)


def basis(v_proj, G_operator, dk, ortho_threshold, first_run=True, Q=None, R=None):
    """
    Generate a complete basis with Chebyshev evolution.

    Parameters
    ----------
    vproj : 2D array
        Collection of filtered vectors.
    G_operator : sparse matrix
        Generator of Chebyshev evolution.
    dk : float
        Steps on Chebyshev evolution before collecting vector.
    ortho_threshold : float
        Threshold for orthogonality condition.
    first_run : boolean
        `True` if it is the first run of Chebyshev evolution before checking degeneracies.
    Q : 2D array
        Q matrix from previous QR decomposition. Only necessary if `first_run=False`.
    R : 2D array
        R matrix from previous QR decomposition. Only necessary if `first_run=False`.
    """
    chebyshev_recursion = chebyshev_recursion_gen(G_operator, v_proj)
    count = 0
    for i in range(G_operator.shape[0]):
        v_n = next(chebyshev_recursion)
        if i == int(i * dk):
            count += 1
            vec = v_n
            if i == 0 and first_run:
                Q, R = scipy.linalg.qr(vec, mode="economic")
            else:
                Q, R = scipy.linalg.qr_insert(
                    Q=Q, R=R, u=vec, k=Q.shape[1], which="col", overwrite_qru=True
                )
                norm = np.arange(1, len(np.diag(R)) + 1)
                ortho_condition = np.abs(np.diag(R) * norm) > ortho_threshold
                if np.invert(ortho_condition).any():
                    indices = np.invert(ortho_condition)
                    return Q[:, indices], R[indices, :][:, indices]
    return Q, R


def index_generator_fn(dk):
    """
    Generate indices of values to store for the eigenvalues-only method.

    Parameters
    ----------
    dk : float
        Steps on Chebyshev evolution before collecting vector.
    """
    items = [-2, -1, 0, 1]
    i = -1
    prev_result = 0
    while True:
        if i < 1:
            prev_result = i + 1
            yield i + 1
        else:
            for item in items:
                result = dk * i + item
                if result == prev_result:
                    continue
                prev_result = result
                yield result
        i += 1


def construct_matrix(k_list_i, k_list_j, storage_list, S_xy):
    """
    Construct matrices with list of S_xy and M_xy.

    Parameters:
    -----------

    k_list_i : integer
        Number of lines.
    k_list_j : integer
        Number of columns.
    storage_list : list
        List of indices.
    S_xy : list
        List of S_xy elements.
    """
    shape_S_xy = S_xy[0].shape
    k_products = np.array(list(it.product(k_list_i, k_list_j)))
    xpy = np.sum(k_products, axis=1).astype(int)
    xmy = np.abs(k_products[:, 0] - k_products[:, 1]).astype(int)

    ind_p = np.searchsorted(storage_list, xpy)
    ind_m = np.searchsorted(storage_list, xmy)

    s_xy = np.asarray(S_xy)
    S = 0.5 * (s_xy[ind_p] + s_xy[ind_m])
    i_size = len(k_list_i)
    j_size = len(k_list_j)
    shape = (i_size, j_size, *shape_S_xy)
    S = np.reshape(S, shape)
    S = np.transpose(S, axes=[2, 0, 3, 4, 1, 5])

    return S


def combine_loops(S_new, S_prev):
    """
    Combine previous and new matrix entries.

    Paramters:
    ----------

    S_new : ndarray
        New array with elements.
    S_prev : ndarray
        Previous array with elements.
    """
    S_conj = np.transpose(S_new[:-1, :, :, :, :, :].conj(), axes=[3, 4, 5, 0, 1, 2])
    n_conj, m_conj = np.prod(S_conj.shape[:3]), np.prod(S_conj.shape[3:])
    S_conj = S_conj.reshape((n_conj, m_conj))
    n_new, m_new = np.prod(S_new.shape[:3]), np.prod(S_new.shape[3:])
    S_new = S_new.reshape((n_new, m_new))
    S_c1 = np.concatenate((S_prev, S_conj), axis=0)
    S_combined = np.concatenate((S_c1, S_new), axis=1)
    return S_combined


def combine_loops_fast(S_diag, S_offdiag, S_prev):
    """
    Combine loops after the two first runs.

    S_diag : ndarray
        Diagonal block of the new matrix.
    S_offdiag : ndarray
        Off-diagonal block of the new matrix.
    S_prev : ndarray
        Previous matrix block.
    """
    n_diag, m_diag = np.prod(S_diag.shape[:3]), np.prod(S_diag.shape[3:])
    S_diag = S_diag.reshape((n_diag, m_diag))
    n_off, m_off = np.prod(S_offdiag.shape[:3]), np.prod(S_offdiag.shape[3:])
    S_offdiag = S_offdiag.reshape((n_off, m_off))
    if S_diag.shape[0] > S_diag.shape[1]:
        S_offdiag = np.concatenate((S_offdiag, S_diag[: -S_diag.shape[1]]), axis=0)
        S_diag = S_diag[-S_diag.shape[1] :]
    S_1 = np.concatenate((S_prev, S_offdiag), axis=1)
    S_2 = np.concatenate((S_offdiag.T.conj(), S_diag), axis=1)
    S_combined = np.concatenate((S_1, S_2), axis=0)
    return S_combined


def eigvals_init(v_proj, G_operator, matrix, dk, ortho_threshold):
    """
    Compute eigenvalues for initial run.

    v_proj : ndarray
        Filtered random vector.
    G_operator : ndarray
        Subspace generator.
    matrix : ndarray
        Matrix to compute eigenvalues.
    dk : integer
        Largest Chebyshev evolution order.
    ortho_threshold : float
        Threshold for orthogonality condition.
    """
    S_xy = []
    matrix_xy = []
    index_generator = index_generator_fn(dk)
    chebyshev_recursion = chebyshev_recursion_gen(G_operator, v_proj)
    storage_list = [next(index_generator)]
    k_list = [0, dk - 1, dk]
    k_latest = 0
    eig_pairs = 1
    v_0 = v_proj[
        np.newaxis,
    ]

    while True:
        v_n = next(chebyshev_recursion)
        if k_latest == storage_list[-1]:
            storage_list.append(next(index_generator))
            S_xy.append(
                np.einsum(
                    "sir,dil->srdl",
                    v_0.conj(),
                    v_n[
                        np.newaxis,
                    ],
                )
            )
            H_v_n = matrix @ v_n
            matrix_xy.append(
                np.einsum(
                    "sir,dil->srdl",
                    v_0.conj(),
                    H_v_n[
                        np.newaxis,
                    ],
                )
            )

        if 2 * eig_pairs * dk + 1 == k_latest:
            S = construct_matrix(k_list, k_list, storage_list, S_xy)
            matrix_proj = construct_matrix(k_list, k_list, storage_list, matrix_xy)
            N = int(np.sqrt(S.size))
            S = S.reshape((N, N))
            matrix_proj = matrix_proj.reshape((N, N))
            q_S, r_S = scipy.linalg.qr(S)
            norm = np.arange(1, len(np.diag(r_S)) + 1)
            ortho_condition = np.abs(np.diag(r_S) * norm) > ortho_threshold
            if np.invert(ortho_condition).any():
                return v_0, k_list, S, matrix_proj, q_S, r_S
            else:
                eig_pairs += 1
                k_list.append(k_list[-1] + dk - 1)
                k_list.append(k_list[-2] + dk)
        k_latest += 1


def eigvals_deg(
    v_prev,
    v_proj,
    k_list,
    S_prev,
    matrix_prev,
    G_operator,
    matrix,
    dk,
    n_evolution=True,
):
    """
    Compute eigenvalues for initial run.

    v_prev : ndarray
        Filtered vectors from previous runs.
    v_proj : ndarray
        Filtered random vector.
    k_list : list
        List of indices from first run.
    G_operator : ndarray
        Subspace generator.
    matrix : ndarray
        Matrix to compute eigenvalues.
    dk : integer
        Largest Chebyshev evolution order.
    n_evolution : bool
    """
    S_xy = []
    matrix_xy = []
    index_generator = index_generator_fn(dk)
    chebyshev_recursion = chebyshev_recursion_gen(G_operator, v_proj)
    storage_list = [next(index_generator)]
    k_latest = 0

    v_0 = v_proj[
        np.newaxis,
    ]
    v_0 = np.concatenate((v_prev, v_0))
    while True:
        v_n = next(chebyshev_recursion)
        if k_latest == storage_list[-1]:
            storage_list.append(next(index_generator))
            S_xy.append(
                np.einsum(
                    "sir,dil->srdl",
                    v_0.conj(),
                    v_n[
                        np.newaxis,
                    ],
                )
            )
            H_v_n = matrix @ v_n
            matrix_xy.append(
                np.einsum(
                    "sir,dil->srdl",
                    v_0.conj(),
                    H_v_n[
                        np.newaxis,
                    ],
                )
            )
        if not n_evolution:
            if 2 * k_list[-1] + 1 == k_latest:
                S = construct_matrix(k_list, k_list, storage_list, S_xy)
                matrix_proj = construct_matrix(k_list, k_list, storage_list, matrix_xy)

                S = combine_loops(S, S_prev)
                matrix_proj = combine_loops(matrix_proj, matrix_prev)
                return v_0, S, matrix_proj
        else:
            if k_list[-1] == k_latest:
                S_xy = np.asarray(S_xy)
                matrix_xy = np.asarray(matrix_xy)
                S_offdiag = construct_matrix(
                    k_list, [0], storage_list, S_xy[:, :n_evolution, :, :, :]
                )
                matrix_proj_offdiag = construct_matrix(
                    k_list, [0], storage_list, matrix_xy[:, :n_evolution, :, :, :]
                )

                S_diag = construct_matrix(
                    [0], [0], storage_list, S_xy[:, n_evolution:, :, :, :]
                )
                matrix_proj_diag = construct_matrix(
                    [0], [0], storage_list, matrix_xy[:, n_evolution:, :, :, :]
                )

                S = combine_loops_fast(S_diag, S_offdiag, S_prev)
                matrix_proj = combine_loops_fast(
                    matrix_proj_diag, matrix_proj_offdiag, matrix_prev
                )
                return v_0, S, matrix_proj
        k_latest += 1


def eigh(
    matrix,
    window_size,
    eps=0.1,
    bounds=None,
    random_vectors=2,
    return_eigenvectors=False,
    filter_order=12,
    error_window=0.1,
    extra_vecs = 0.3,
):
    """
    Find the eigendecomposition within the given spectral bounds of a given matrix.

    Parameters
    ----------
    matrix : 2D array or sparse matrix
        Real of complex Hermitian matrix to diagonalize.
    window_size : float
        Energy window around zero for which to solve the eigenproblem.
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
    error_window : float
        The fraction by which to expands the window size to account for errors.
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("expected square matrix (shape=%s)" % (matrix.shape,))
    if filter_order <= 0:
        raise ValueError("filter_order must be greater than 0.")
    if eps < 0:
        raise ValueError("eps must be greater than or equal to 0.")
    if random_vectors <= 0:
        raise ValueError("random_vectors must be greater than 0.")
    if error_window < 0:
        raise ValueError("error_window must be greater than or equal to 0.")

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
                "obtain continue."
            )

        bounds = [lmin, lmax]

    if lmin > 0:
        raise ValueError("Lower bound of the spectrum must be negative.")
    if lmax < 0:
        raise ValueError("Upper bound of the spectrum must be positive.")

    a = window_size * (1 + error_window)
    if a > min(abs(lmin), abs(lmax)):
        raise ValueError("a must be smaller than spectrum bounds.")

    ortho_threshold = 10 * np.sqrt(matrix.shape[0]) * np.exp(-2 * filter_order)
    if ortho_threshold < 10 * np.finfo(float).eps:
        warnings.warn("Results limited by numerical precision.")
        ortho_threshold = 10 * np.finfo(float).eps
    if ortho_threshold > 1e-6:
        warnings.warn("Filter order is too small. Fixing it to avoid errors.")
        filter_order = np.ceil(-0.5 * np.log(1e-7 / np.sqrt(matrix.shape[0])))

    Emin = bounds[0] * (1 + eps)
    Emax = bounds[1] * (1 + eps)
    E0 = (Emax - Emin) / 2
    Ec = (Emax + Emin) / 2
    G_operator = (matrix - eye(matrix.shape[0]) * Ec) / E0

    Emax = np.max(np.abs(bounds)) * (1 + eps)
    E0 = (Emax**2 - a**2) / 2
    Ec = (Emax**2 + a**2) / 2
    F_operator = (matrix @ matrix - eye(matrix.shape[0]) * Ec) / E0

    def get_filtered_vector():
        v_rand = 2 * (
            np.random.rand(matrix.shape[0], random_vectors)
            + np.random.rand(matrix.shape[0], random_vectors) * 1j
            - 0.5 * (1 + 1j)
        )
        v_rand = v_rand / np.linalg.norm(v_rand, axis=0)
        K_max = int(filter_order * np.max(np.abs(bounds)) / a)
        vec = low_E_filter(v_rand, F_operator, K_max)
        return vec / np.linalg.norm(vec, axis=0)

    a_r = a / np.max(np.abs(bounds))
    dk = ceil(np.pi / a_r)

    if return_eigenvectors:
        # First run
        Q, R = basis(
            v_proj=get_filtered_vector(),
            G_operator=G_operator,
            dk=dk,
            ortho_threshold=ortho_threshold,
        )
        # Second run
        Qi, Ri = basis(
            v_proj=get_filtered_vector(),
            G_operator=G_operator,
            dk=dk,
            ortho_threshold=ortho_threshold,
            Q=Q,
            R=R,
            first_run=False,
        )
        # Other runs to solve higher degeneracies
        while Q.shape[1] < Qi.shape[1]:
            Q, R = Qi, Ri
            Qi, Ri = basis(
                v_proj=get_filtered_vector(),
                G_operator=G_operator,
                dk=dk,
                ortho_threshold=ortho_threshold,
                Q=Q,
                R=R,
                first_run=False,
            )

        matrix_proj = Q.conj().T @ matrix @ Q
        eigvals, eigvecs = scipy.linalg.eigh(matrix_proj)
        eigvecs = (Q @ eigvecs).T
        window_args = np.abs(eigvals) < window_size
        return eigvals[window_args], eigvecs[window_args, :]

    else:
        N_loop = 0
        n_evolution = False
        while True:
            v_proj = get_filtered_vector()
            if N_loop == 0:
                v_0, k_list, S, matrix_proj, q_S, r_S = eigvals_init(
                    v_proj, G_operator, matrix, dk, ortho_threshold
                )
                norm = np.arange(1, len(np.diag(r_S)) + 1)
                N_H_prev = sum(np.abs(np.diag(r_S) * norm) > ortho_threshold)
                new_vals = N_H_prev
            else:
                if new_vals <= random_vectors and not n_evolution:
                    n_evolution = N_loop
                v_0, S, matrix_proj = eigvals_deg(
                    v_0,
                    v_proj,
                    k_list,
                    S,
                    matrix_proj,
                    G_operator,
                    matrix,
                    dk,
                    n_evolution,
                )
                q_S, r_S = scipy.linalg.qr_insert(
                    Q=q_S,
                    R=r_S,
                    u=S[: q_S.shape[0], q_S.shape[1] :],
                    k=q_S.shape[1],
                    which="col",
                )
                q_S, r_S = scipy.linalg.qr_insert(
                    Q=q_S,
                    R=r_S,
                    u=S[q_S.shape[0] :, :],
                    k=q_S.shape[0],
                    which="row",
                )
                norm = np.arange(1, len(np.diag(r_S)) + 1)
                N_H_cur = sum(np.abs(np.diag(r_S) * norm) > ortho_threshold)
                new_vals = N_H_cur - N_H_prev
                if new_vals > 0:
                    N_H_prev = N_H_cur
                else:
                    diagS = np.diag(np.diag(S))
                    S = S - diagS + diagS.real
                    H_red = svd_decomposition(S, matrix_proj)
                    eigvals = scipy.linalg.eigvalsh(H_red)
                    window_args = np.abs(eigvals) < window_size
                    return eigvals[window_args]
            N_loop += 1