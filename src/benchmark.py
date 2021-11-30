import kwant
import kwant.linalg.mumps as mumps
from pyDACP import core, chebyshev
import time
from scipy.linalg import eigvalsh
import scipy.sparse.linalg as sla
from scipy.sparse import identity
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

# +
t = 1

def make_syst(N=100, dimension=2):

    if dimension==2:
        L=np.sqrt(N)
        lat = kwant.lattice.square(a=1, norbs=1)
        # Define 2D shape
        def shape(pos):
            (x, y) = pos
            return (-L < x < L) and (-L < y < L)
    elif dimension==3:
        L=np.cbrt(N)
        lat = kwant.lattice.cubic(a=1, norbs=1)
        # Define 3D shape
        def shape(pos):
            (x, y, z) = pos
            return (-L < x < L) and (-L < y < L)  and (-L < z < L)

    syst = kwant.Builder()

    def onsite(site, seed):
        delta_mu = kwant.digest.uniform(repr(site.pos) + str(seed)) - 0.5
        return delta_mu

    if dimension==2:
        syst[lat.shape(shape, (0, 0))] = onsite
    elif dimension==3:
        syst[lat.shape(shape, (0, 0, 0))] = onsite
    syst[lat.neighbors()] = -t

    return syst.finalized()


# -

def sparse_diag(matrix, k, sigma, **kwargs):
    """Call sla.eigsh with mumps support.

    Please see scipy.sparse.linalg.eigsh for documentation.
    """
    class LuInv(sla.LinearOperator):
        def __init__(self, A):
            inst = mumps.MUMPSContext()
            inst.analyze(A, ordering='pord')
            inst.factor(A)
            self.solve = inst.solve
            sla.LinearOperator.__init__(self, A.dtype, A.shape)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    opinv = LuInv(matrix - sigma * identity(matrix.shape[0]))
    return sla.eigsh(matrix, k, sigma=sigma, OPinv=opinv, **kwargs)


def benchmark(N, seed, dimension):
    syst = make_syst(N=N, dimension=dimension)
    H = syst.hamiltonian_submatrix(params=dict(seed=seed), sparse=True)
    dacp = core.DACP_reduction(H, a=0.2, eps=0.05)
    d = dacp.estimate_subspace_dimenstion()

    def sparse_benchmark():
        _ = sparse_diag(H, sigma=0, k=int(
            dacp.sampling_subspace*d), which='LM', return_eigenvectors=False)
    start_time = time.time()
    sparse_memory=memory_usage(sparse_benchmark, max_usage=True)
    sparse_time=time.time() - start_time

    sparse_data = [sparse_time, sparse_memory]

    def dacp_benchmark():
        dacp = core.DACP_reduction(H, a=0.2, eps=0.05)
        _ = eigvalsh(dacp.get_subspace_matrix())
    start_time = time.time()
    dacp_memory=memory_usage(dacp_benchmark, max_usage=True)
    dacp_time=time.time() - start_time

    dacp_data = [dacp_time, dacp_memory]

    return sparse_data, dacp_data


benchmark(1e3, 1, 2)

benchmark(5e2, 1, 3)

seeds=np.linspace(1, 10, 10)
dimensions=[2, 3]
Ns=np.linspace(1e2, 5e3)


