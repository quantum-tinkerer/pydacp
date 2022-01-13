import numpy as np
from scipy.linalg import qr, qr_insert
import itertools as it


def chebyshev_recursion_gen(matrix, v_proj):
    order = 0
    while True:
        if order == 0:
            v_n = v_proj
        elif order == 1:
            v_nm1 = v_n
            v_n = matrix.dot(v_nm1)
        else:
            v_np1 = 2 * matrix.dot(v_n) - v_nm1
            v_nm1 = v_n
            v_n = v_np1
        order += 1
        yield v_n


def low_E_filter(v_proj, matrix, k):
    chebyshev_recursion = chebyshev_recursion_gen(matrix, v_proj)
    for i in range(k + 1):
        v_n = next(chebyshev_recursion)
    return v_n/np.linalg.norm(v_n, axis=0)


def basis(v_proj, matrix, dk, first_run=True, Q=None, R=None):
    chebyshev_recursion = chebyshev_recursion_gen(matrix, v_proj)
    for i in range(matrix.shape[0]):
        v_n = next(chebyshev_recursion)
        if i == int(i * dk):
            vec = v_n / np.linalg.norm(v_n, axis=0)
            if i == 0 and first_run:
                Q, R = qr(vec, mode="economic")
            else:
                Q, R = qr_insert(
                    Q=Q, R=R, u=vec, k=Q.shape[1], which="col", overwrite_qru=True
                )
                ortho_condition = np.abs(np.diag(R)) < 1e-9
                if ortho_condition.any():
                    indices = np.invert(ortho_condition)
                    return Q[:, indices], R[indices, :][:, indices]
    return Q, R


def index_generator_fn(dk):
    items = [-2, -1, 0, 1]
    i = -1
    prev_result = 0
    while True:
        if i < 1:
            prev_result = i + 1
            yield i + 1
        else:
            for item in items:
                result = dk*i+item
                if result == prev_result:
                    continue
                prev_result = result
                yield result
        i += 1


def basis_no_store(v_proj, matrix, H, dk, random_vectors):
    S_xy = []
    H_xy = []
    index_generator = index_generator_fn(dk)
    chebyshev_recursion = chebyshev_recursion_gen(matrix, v_proj)
    storage_list = [next(index_generator)]
    k_list = [0, dk-1, dk]
    k_latest = 0
    index = 1
    while True:
        v_n = next(chebyshev_recursion)
        if k_latest == storage_list[-1]:
            storage_list.append(next(index_generator))
            S_xy.append(v_proj.conj().T @ v_n)
            H_xy.append(v_proj.conj().T @ H @ v_n)
        if 2*index*dk + 1 == k_latest: #not sure whether to +1 or not so delete maybe sometime future
            k_products = np.array(list(it.product(k_list, k_list)))
            xpy = np.sum(k_products, axis=1).astype(int)
            xmy = np.abs(k_products[:, 0] - k_products[:, 1]).astype(int)

            ind_p = np.searchsorted(storage_list, xpy)
            ind_m = np.searchsorted(storage_list, xmy)

            s_xy = np.asarray(S_xy)
            h_xy = np.asarray(H_xy)
            S = 0.5 * (s_xy[ind_p] + s_xy[ind_m])
            matrix_proj = 0.5 * (h_xy[ind_p] + h_xy[ind_m])
            m = len(k_list)

            S = np.reshape(S, (m, m, random_vectors, random_vectors))
            S = np.hstack(np.hstack(S))
            matrix_proj = np.reshape(matrix_proj, (m, m, random_vectors, random_vectors))
            matrix_proj = np.hstack(np.hstack(matrix_proj))
            q_S, r_S = qr(S)
            ortho_condition = np.abs(np.diag(r_S)) < 1e-9
            if ortho_condition.any():
                indices = np.invert(ortho_condition)
                return S[indices, :][:, indices], matrix_proj[indices, :][:, indices]
            else:
                index += 1
                k_list.append(k_list[-1]+dk-1)
                k_list.append(k_list[-2]+dk)
        k_latest += 1
