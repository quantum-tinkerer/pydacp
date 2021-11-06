import numpy as np

def low_E_filter(ψ_rand, matrix, k):
    for i in range(k+1):
        if i == 0:
            ψ_n = ψ_rand
            continue

        elif i == 1:
            ψ_nm1 = ψ_n
            ψ_n = matrix @ ψ_nm1
            continue
        else:
            ψ_np1 = 2*matrix @ ψ_n - ψ_nm1
            ψ_nm1 = ψ_n
            ψ_n = ψ_np1
    return ψ_n/np.linalg.norm(ψ_n)


def basis(ψ_proj, matrix, indices):
    ψ_basis = []
    # TODO: If k is too large, the norms of the vectors are from some large order.
    k = indices[-1]
    for i in range(k+1):
        if i == 0:
            ψ_n = ψ_proj
            continue
        elif i == 1:
            ψ_nm1 = ψ_n
            ψ_n = matrix @ ψ_nm1
            continue
        else:
            ψ_np1 = 2 * matrix @ ψ_n - ψ_nm1
            ψ_nm1 = ψ_n
            ψ_n = ψ_np1
        if i in indices:
            ψ_basis.append(ψ_n/ np.linalg.norm(ψ_n))
    return np.asarray(ψ_basis)