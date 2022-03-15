import numpy as np
from scipy.linalg import qr, qr_insert, eigvalsh
import itertools as it
import matplotlib.pyplot as plt


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


def basis(v_proj, G_operator, dk, first_run=True, Q=None, R=None):
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
    first_run : boolean
        `True` if it is the first run of Chebyshev evolution before checking degeneracies.
    Q : 2D array
        Q matrix from previous QR decomposition. Only necessary if `first_run=False`.
    R : 2D array
        R matrix from previous QR decomposition. Only necessary if `first_run=False`.
    """
    chebyshev_recursion = chebyshev_recursion_gen(G_operator, v_proj)
    for i in range(G_operator.shape[0]):
        v_n = next(chebyshev_recursion)
        if i == int(i * dk):
            vec = v_n / np.linalg.norm(v_n, axis=0)
            if i == 0 and first_run:
                Q, R = qr(vec, mode="economic")
            else:
                Q, R = qr_insert(
                    Q=Q, R=R, u=vec, k=Q.shape[1], which="col", overwrite_qru=True
                )
                ortho_condition = np.abs(np.diag(R)) < 1e-12
                if ortho_condition.any():
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
    shape = (
        i_size,
        j_size,
        *shape_S_xy
    )
    S = np.reshape(S, shape)
    S = np.transpose(S, axes=[0, 2, 3, 1, 4, 5])

    return S


def combine_loops(S_new, S_prev):
    S_conj = np.transpose(S_new[:, :-1, :, :, :, :].conj(), axes=[3, 4, 5, 0, 1, 2])
    S_c1 = np.concatenate((S_prev, S_conj), axis=1)
    S_combined = np.concatenate((S_c1, S_new), axis=-2)
    return S_combined


def eigvals_init(v_proj, G_operator, matrix, dk):
    S_xy = []
    matrix_xy = []
    index_generator = index_generator_fn(dk)
    chebyshev_recursion = chebyshev_recursion_gen(G_operator, v_proj)
    storage_list = [next(index_generator)]
    k_list = [0, dk-1, dk]
    k_latest = 0
    eig_pairs = 1
    v_0 = v_proj[np.newaxis,]

    while True:
        v_n = next(chebyshev_recursion)
        if k_latest == storage_list[-1]:
            storage_list.append(next(index_generator))
            S_xy.append(np.einsum('sir,dil->srdl', v_0.conj(), v_n[np.newaxis,]))
            H_v_n = matrix@v_n
            matrix_xy.append(np.einsum('sir,dil->srdl', v_0.conj(), H_v_n[np.newaxis,]))

        if 2*eig_pairs*dk + 1 == k_latest:
            S = construct_matrix(k_list, k_list, storage_list, S_xy)
            matrix_proj = construct_matrix(k_list, k_list, storage_list, matrix_xy)
            N = int(np.sqrt(S.size))
            S_sq = S.reshape((N, N))
            matrix_proj_sq = matrix_proj.reshape((N,N))
            q_S, r_S = qr(S_sq)
            ortho_condition = np.abs(np.diag(r_S)) < 1e-12
            if ortho_condition.any():
                return v_0, k_list, S, matrix_proj
            else:
                eig_pairs += 1
                k_list.append(k_list[-1]+dk-1)
                k_list.append(k_list[-2]+dk)
        k_latest += 1


def eigvals_deg(v_prev, v_proj, k_list, S_prev, matrix_prev, G_operator, matrix, dk):
    S_xy = []
    matrix_xy = []
    index_generator = index_generator_fn(dk)
    chebyshev_recursion = chebyshev_recursion_gen(G_operator, v_proj)
    storage_list = [next(index_generator)]
    k_latest = 0

    v_0 = v_proj[np.newaxis,]
    v_0 = np.concatenate((v_prev, v_0))
    while True:
        v_n = next(chebyshev_recursion)
        if k_latest == storage_list[-1]:
            storage_list.append(next(index_generator))
            S_xy.append(np.einsum('sir,dil->srdl', v_0.conj(), v_n[np.newaxis,]))
            H_v_n = matrix@v_n
            matrix_xy.append(np.einsum('sir,dil->srdl', v_0.conj(), H_v_n[np.newaxis,]))

        if 2*k_list[-1]+1 == k_latest:
            S = construct_matrix(k_list, k_list, storage_list, S_xy)
            matrix_proj = construct_matrix(k_list, k_list, storage_list, matrix_xy)

            S = combine_loops(S, S_prev)
            matrix_proj = combine_loops(matrix_proj, matrix_prev)
            return v_0, S, matrix_proj
        k_latest += 1