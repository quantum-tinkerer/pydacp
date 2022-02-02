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
    # print(v_proj.shape)
    chebyshev_recursion = chebyshev_recursion_gen(G_operator, v_proj)
    storage_list = [next(index_generator)]
    k_list = [0, dk - 1, dk]
    k_latest = 0
    index = 1
    while True:
        v_n = next(chebyshev_recursion)
        if k_latest == storage_list[-1]:
            storage_list.append(next(index_generator))
            S_xy.append(v_proj.conj().T @ v_n)
            matrix_xy.append(v_proj.conj().T @ matrix @ v_n)
        if 2 * index * dk + 1 == k_latest:
            # not sure whether to +1 or not so delete maybe sometime future
            k_products = np.array(list(it.product(k_list, k_list)))
            xpy = np.sum(k_products, axis=1).astype(int)
            xmy = np.abs(k_products[:, 0] - k_products[:, 1]).astype(int)

            ind_p = np.searchsorted(storage_list, xpy)
            ind_m = np.searchsorted(storage_list, xmy)

            s_xy = np.asarray(S_xy)
            m_xy = np.asarray(matrix_xy)
            S_diag = 0.5 * (s_xy[ind_p] + s_xy[ind_m])
            matrix_proj_diag = 0.5 * (m_xy[ind_p] + m_xy[ind_m])
            m = len(k_list)

            shape_diag = (
                m,
                m,
                random_vectors,
                random_vectors,
            )

            S_diag = np.reshape(S_diag, shape_diag)
            S_diag = np.concatenate(
                np.concatenate(
                    S_diag,
                    axis=1,
                ),
                axis=1,
            )
            norms_diag = 1 / np.sqrt(np.diag(S_diag))
            norms = np.outer(norms_diag, norms_diag)
            S_diag = np.multiply(S_diag, norms)
            matrix_proj_diag = np.reshape(matrix_proj_diag, shape_diag)
            matrix_proj_diag = np.concatenate(
                np.concatenate(
                    matrix_proj_diag,
                    axis=1,
                ),
                axis=1,
            )
            matrix_proj_diag = np.multiply(matrix_proj_diag, norms)

            if first_run:
                S = S_diag
                matrix_proj = matrix_proj_diag

            else:
                # Construct block off-diagonal

                # This part can be made much faster using the Chebyshev relation:
                # T_i(x) * T_j(x) = 0.5 * (T_{i+j}(x) + T_{|i-j|}(x))
                # And probably can reuse values already computed in the previous step.

                # This part goes as follows:
                # We want to compute Hij and Sij.
                # i goes from 1 to n
                # j goes from 1 to k_list[-1] / dk
                # As you can see, n is different for different iterations.
                # Obtain list of indices used to construct S_diag

                Sij_prev = []
                matrix_ij_prev = []
                for i, n in enumerate(indices_prev):
                    Sij_prev_n = []
                    matrix_ij_prev_n = []
                    k_prev_1 = np.arange(1, n + 1, 1) * dk
                    k_prev_2 = k_prev_1 - 1
                    k_prev_list = np.unique(np.concatenate([[0], k_prev_1, k_prev_2]))
                    # Chebolve random vectors from previous run
                    chebyshev_recursion_prev_off = chebyshev_recursion_gen(
                        G_operator, v_prev[i]
                    )
                    for k_prev in range(k_prev_list[-1]+1):
                        # Generate next vector in the Chebolution
                        v_prev_off = next(chebyshev_recursion_prev_off)
                        # Loop over j's
                        Sj_prev_n = []
                        matrix_j_prev_n = []
                        # This is storing all i's
                        if k_prev in k_prev_list:
                            # Chebolve random vectors from this run
                            chebyshev_recursion_proj_off = chebyshev_recursion_gen(
                                G_operator, v_proj
                            )
                            for k_proj in range(k_list[-1]+1):
                                # Generate next vector in the Chebolution
                                v_proj_off = next(chebyshev_recursion_proj_off)
                                # This is storing all j's
                                if k_proj in k_list:
                                    # Normalize
                                    v_a = v_proj_off / np.linalg.norm(
                                        v_proj_off, axis=0
                                    )
                                    v_b = v_prev_off / np.linalg.norm(
                                        v_prev_off, axis=0
                                    )
                                    Sj_prev_n.append(v_b.conj().T @ v_a)
                                    matrix_j_prev_n.append(v_b.conj().T @ matrix @ v_a)
                            Sij_prev_n.append(np.stack(Sj_prev_n))
                            matrix_ij_prev_n.append(np.stack(matrix_j_prev_n))
                    Sij_prev.append(np.stack(Sij_prev_n))
                    matrix_ij_prev.append(np.stack(matrix_ij_prev_n))

                S_off = np.vstack(Sij_prev)
                S_off = np.concatenate(
                    np.concatenate(
                        S_off,
                        axis=1,
                    ),
                    axis=1,
                )
                matrix_proj_off = np.vstack(matrix_ij_prev)
                matrix_proj_off = np.concatenate(
                    np.concatenate(
                        matrix_proj_off,
                        axis=1,
                    ),
                    axis=1,
                )

                # Glue everything
                S1 = np.hstack([S_prev, S_off])
                S2 = np.hstack([S_off.conj().T, S_diag])
                S = np.vstack([S1, S2])
                M1 = np.hstack([matrix_prev, matrix_proj_off])
                M2 = np.hstack([matrix_proj_off.conj().T, matrix_proj_diag])
                matrix_proj = np.vstack([M1, M2])

            # Perform QR orthogonalization of overlap matrix
            # q_S, r_S = qr(S)
            # ortho_condition = np.abs(np.diag(r_S)) < 1e-6
            s = eigvalsh(S)
            indx = np.abs(s) > 1e-3
            dim = sum(indx)
            # if ortho_condition.any():
            if dim < S.shape[0]:
                # qr_idx = np.invert(ortho_condition)
                # dim = sum(qr_idx)
                print("Found " + str(dim) + " vectors so far.")
                if first_run:
                    indices_chebolution = [index]
                else:
                    indices_chebolution = indices_prev.copy()
                    indices_chebolution.append(index)
                return S, matrix_proj, dim, indices_chebolution
            else:
                # S_prev, matrix_prev = S, matrix_proj
                index += 1
                k_list.append(k_list[-1] + dk - 1)
                k_list.append(k_list[-2] + dk)
        k_latest += 1
