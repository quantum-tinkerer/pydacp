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


def construct_matrix(k_list_i, k_list_j, storage_list, S_xy, random_vectors):
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
        random_vectors,
        random_vectors,
    )
    S = np.reshape(S, shape)
    return S


def basis_no_store(
    v_proj,
    G_operator,
    matrix,
    dk,
    random_vectors,
    v_prev=None,
    first_run=True,
    S_prev=None,
    matrix_prev=None,
    indices_prev=None,
):
    """
    Generate a complete basis with Chebyshev evolution.

    Parameters
    ----------
    vproj : 2D array
        Collection of filtered vectors.
    G_operator : sparse matrix
        Generator of Chebyshev evolution.
    H : sparse matrix
        Initial matrix.
    dk : float
        Steps on Chebyshev evolution before collecting vector.
    random_vectors : int
        Number of random vectors.x
    """
    S_xy = []
    matrix_xy = []
    index_generator = index_generator_fn(dk)
    chebyshev_recursion = chebyshev_recursion_gen(G_operator, v_proj)
    storage_list = [next(index_generator)]
    k_list = [0, dk - 1, dk]
    k_latest = 0
    index = 1
    if not first_run:
        S_xy_off = []
        matrix_xy_off = []
    while True:
        v_n = next(chebyshev_recursion)
        if first_run:
            if k_latest == storage_list[-1]:
                storage_list.append(next(index_generator))
                S_xy.append(v_proj.conj().T @ v_n)
                matrix_xy.append(v_proj.conj().T @ matrix @ v_n)
            if 2 * index * dk + 1 == k_latest:
                S = construct_matrix(
                    k_list,
                    k_list,
                    storage_list,
                    S_xy,
                    random_vectors
                )
                matrix_proj = construct_matrix(
                    k_list,
                    k_list,
                    storage_list,
                    matrix_xy,
                    random_vectors
                )
                S = np.concatenate(
                    np.concatenate(
                        S,
                        axis=1,
                    ),
                    axis=1,
                )
                matrix_proj = np.concatenate(
                    np.concatenate(
                        matrix_proj,
                        axis=1,
                    ),
                    axis=1,
                )
                # Perform QR orthogonalization of overlap matrix
                # q_S, r_S = qr(S_diag)
                # ortho_condition = np.abs(np.diag(r_S)) < 1e-9
                s = eigvalsh(S)
                indx = s > 1e-9
                dim = sum(indx)
                # if ortho_condition.any():
                if dim < S.shape[0]:
                    # q_S, r_S = qr(S)
                    # indx = np.abs(np.diag(r_S)) < 1e-9
                    # qr_idx = np.invert(ortho_condition)
                    # dim = sum(qr_idx)
                    print("Found " + str(dim) + " eigenvalues so far.")
                    indices_chebolution = [index]
                    return S, matrix_proj, dim, indices_chebolution
                else:
                    # S_prev, matrix_prev = S, matrix_proj
                    index += 1
                    k_list.append(k_list[-1] + dk - 1)
                    k_list.append(k_list[-2] + dk)
            k_latest += 1

        else:
            if k_latest == storage_list[-1]:
                storage_list.append(next(index_generator))
                S_xy.append(v_proj.conj().T @ v_n)
                matrix_xy.append(v_proj.conj().T @ matrix @ v_n)
                S_xy_off_n = []
                matrix_xy_off_n = []
                for v_prev_n in v_prev:
                    S_xy_off.append(v_prev_n.conj().T @ v_n)
                    matrix_xy_off.append(v_prev_n.conj().T @ matrix @ v_n)
                # S_xy_off.append(np.stack(S_xy_off_n))
                # matrix_xy_off.append(np.stack(matrix_xy_off_n))
            if k_latest == 2 * np.max([*indices_prev, index]) * dk + 1:
                print(np.asarray(S_xy_off).shape)
                S_diag = construct_matrix(
                    k_list, k_list, storage_list, S_xy, random_vectors
                )
                S_diag = np.concatenate(
                    np.concatenate(
                        S_diag,
                        axis=1,
                    ),
                    axis=1,
                )
                matrix_proj_diag = construct_matrix(
                    k_list, k_list, storage_list, matrix_xy, random_vectors
                )
                matrix_proj_diag = np.concatenate(
                    np.concatenate(
                        matrix_proj_diag,
                        axis=1,
                    ),
                    axis=1,
                )
                S_off_n = []
                matrix_off_n = []
                for i, n in enumerate(indices_prev):
                    k_prev_n = np.arange(1, n + 1, 1) * dk
                    k_list_prev = np.unique(
                        np.concatenate([[0], k_prev_n, k_prev_n - 1])
                    )
                    S_off_n.append(
                        construct_matrix(
                            k_list_prev,
                            k_list,
                            storage_list,
                            S_xy_off,
                            random_vectors
                        )
                    )
                    matrix_off_n.append(
                        construct_matrix(
                            k_list_prev,
                            k_list,
                            storage_list,
                            matrix_xy_off,
                            random_vectors,
                        )
                    )

                S_off = np.asarray(np.vstack(S_off_n))
                S_off = np.concatenate(
                    np.concatenate(
                        S_off,
                        axis=1,
                    ),
                    axis=1,
                )
                matrix_proj_off = np.asarray(np.vstack(matrix_off_n))
                matrix_proj_off = np.concatenate(
                    np.concatenate(
                        matrix_proj_off,
                        axis=1,
                    ),
                    axis=1,
                )

                S = np.block(
                    [[S_prev, S_off],
                     [S_off.conj().T, S_diag]]
                )
                matrix_proj = np.block(
                    [
                        [matrix_prev, matrix_proj_off],
                        [matrix_proj_off.conj().T, matrix_proj_diag],
                    ]
                )

                # Perform QR orthogonalization of overlap matrix
                # q_S, r_S = qr(S_diag)
                # ortho_condition = np.abs(np.diag(r_S)) < 1e-9
                # s = eigvalsh(S_diag)
                # indx = s > 1e-6
                # dim_diag = sum(indx)
                # if ortho_condition.any():
                # print(dim_diag, S_diag.shape[0])
                # if dim_diag < S_diag.shape[0]:
                    # q_S, r_S = qr(S)
                    # indx = np.abs(np.diag(r_S)) < 1e-9
                    # qr_idx = np.invert(ortho_condition)
                    # dim = sum(qr_idx)
                # Normalization
                Si = S
                norms = 1 / np.sqrt(np.diag(Si))
                norms = np.outer(norms, norms)
                Si = np.multiply(Si, norms)
                s = eigvalsh(Si)
                indx = s > 1e-3
                dim = sum(indx)
                print("Found " + str(dim) + " eigenvalues so far.")
                indices_chebolution = indices_prev.copy()
                indices_chebolution.append(index)
                return S, matrix_proj, dim, indices_chebolution
            #     else:
            #         if index >= np.max(indices_prev):
            #             return S, matrix_proj, dim, indices_chebolution
            #         # S_prev, matrix_prev = S, matrix_proj
            #         else:
            #             index += 1
            #             k_list.append(k_list[-1] + dk - 1)
            #             k_list.append(k_list[-2] + dk)
            k_latest += 1
