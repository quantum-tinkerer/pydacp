import kwant
import kwant.linalg.mumps as mumps
from pyDACP import core, chebyshev
import time
from scipy.linalg import eigvalsh, eigh
import scipy.sparse.linalg as sla
from scipy.sparse import identity
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from itertools import product
from tqdm import tqdm
import itertools as it
import xarray as xr

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
        return delta_mu - 2

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
    a=0.1

    def cheb_benchmark():
        dacp = core.DACP_reduction(H, a=a, eps=0.05, return_eigenvectors=True, chebolution=True)
        _, _ = eigh(dacp.get_subspace_matrix()[0])
    start_time = time.time()
    cheb_memory=memory_usage(cheb_benchmark, max_usage=True)
    cheb_time=time.time() - start_time

    cheb_data = [cheb_time, cheb_memory]

    def dacp_benchmark():
        dacp = core.DACP_reduction(H, a=a, eps=0.05, return_eigenvectors=True, chebolution=False)
        _, _ = eigh(dacp.get_subspace_matrix()[0])
    start_time = time.time()
    dacp_memory=memory_usage(dacp_benchmark, max_usage=True)
    dacp_time=time.time() - start_time

    dacp_data = [dacp_time, dacp_memory]

    return cheb_data, dacp_data


# +
params = {
    'N' : [10**i for i in np.linspace(3, 4, 5, endpoint=True)],
    'seeds' : [1],
    'dimensions' : [2]#[2,3]
}

values = list(params.values())
args = np.array(list(it.product(*values)))

args = tqdm(args)

def wrapped_fn(args):
    a, b = benchmark(*args)
    return *a, *b

result = list(map(wrapped_fn, args))
# -

shapes = [len(values[i]) for i in range(len(values))]
shapes = [*shapes, 4]
shaped_data = np.reshape(result, shapes)

shaped_data = np.reshape(result, shapes)
da = xr.DataArray(
    data=shaped_data,
    dims=[*params.keys(), 'output'],
    coords={
        **params,
        'output': ['chebtime', 'chebmem', 'dacptime', 'dacpmem']
    }
)

da_mean=da.mean(dim='seeds')

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 15

da_mean.sel(dimensions=2).sel(output=['dacptime', 'chebtime']).plot(hue='output')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('time_2d_cheb.png')
plt.show()

da_mean.sel(dimensions=2).sel(output=['dacpmem', 'chebmem']).plot(hue='output')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('mem_2d_cheb.png')
plt.show()

da_mean.sel(dimensions=3).sel(output=['dacptime', 'chebtime']).plot(hue='output')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('time_3d_cheb.png')
plt.show()

da_mean.sel(dimensions=3).sel(output=['dacpmem', 'chebmem']).plot(hue='output')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('mem_3d_cheb.png')
plt.show()
