import kwant
import kwant.linalg.mumps as mumps
from pyDACP import core, chebyshev
import time
from scipy.linalg import eigvalsh, eigh
import scipy.sparse.linalg as sla
from scipy.sparse import identity, diags
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
    elif dimension==1:
        L=N
        lat = kwant.lattice.chain(a=1, norbs=1)
        # Define 1D shape
        def shape(pos):
            x = pos[0]
            return (-L < x < L)

    syst = kwant.Builder()

    def onsite(site, seed):
        delta_mu = kwant.digest.uniform(repr(site.pos) + str(seed)) - 0.5
        return delta_mu - 2

    if dimension==2:
        syst[lat.shape(shape, (0, 0))] = onsite
    elif dimension==3:
        syst[lat.shape(shape, (0, 0, 0))] = onsite
    elif dimension==1:
        syst[lat.shape(shape, (0,))] = onsite
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
    lmax = float(sla.eigsh(H, k=1, which="LA", return_eigenvectors=False))
    lmin = float(sla.eigsh(H, k=1, which="SA", return_eigenvectors=False))
    a = np.abs(lmax - lmin) / 2 * N ** (1 / 2) / N

    def dacp_benchmark():
        eigs = core.dacp_eig(
            H,
            a=a,
            eps=0.05,
            return_eigenvectors=False,
            filter_order=15,
            random_vectors=5,
        )
        return len(eigs)

    start_time = time.time()
    dacp_memory, k = memory_usage(dacp_benchmark, max_usage=True, retval=True)
    dacp_time = time.time() - start_time

    dacp_data = [dacp_time, dacp_memory]

    return dacp_data


# +
params = {
    'N' : [10**i for i in np.linspace(3, 5, 5, endpoint=True)],
    'seeds' : [1],#, 2, 3],
    'dimensions' : [1]
}

values = list(params.values())
args = np.array(list(it.product(*values)))

args = tqdm(args)

def wrapped_fn(args):
    a = benchmark(*args)
    return a

result = list(map(wrapped_fn, args))
# -

shapes = [len(values[i]) for i in range(len(values))]
shapes = [*shapes, 2]
shaped_data = np.reshape(result, shapes)

shaped_data = np.reshape(result, shapes)
da = xr.DataArray(
    data=shaped_data,
    dims=[*params.keys(), 'output'],
    coords={
        **params,
        'output': ['dacptime', 'dacpmem']
    }
)

da.to_netcdf('./benchmark_data/data_dacp_eigsonly_1d.nc')

da_mean=da.mean(dim='seeds')

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 15

from scipy.stats import linregress


def slope(dim, output):
    print(
        linregress(
            np.log(da_mean.N.values),
            np.log(da_mean.sel(output=output, dimensions=dim).values)
        )[0]
    )


slope(dim=1, output='dacptime')

# +
# slope(dim=2, output='dacptime')

# +
# slope(dim=3, output='dacptime')
# slope(dim=3, output='sparsetime')
# -

slope(dim=1, output='dacpmem')

# +
# slope(dim=2, output='dacpmem')

# +
# slope(dim=3, output='dacpmem')
# slope(dim=3, output='sparsemem')
# -

from sys import getsizeof

getsizeof(np.random.rand(int(1e6)) + 1j * np.random.rand(int(1e6))) / 1e6 * 1000

(da_mean.sel(dimensions=1).sel(output=['dacptime'])/3600).plot(hue='output', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.ylabel(r'$\mathrm{Time\ [hours]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
plt.savefig('time_1d_eigsonly_dacponly.png')
plt.show()

(da_mean.sel(dimensions=1).sel(output=['dacpmem'])/1000).plot(hue='output', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathrm{Memory\ [GB]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
plt.tight_layout()
plt.savefig('mem_1d_eigsonly_dacponly.png')
plt.show()

# +
# da_mean.sel(dimensions=2).sel(output=['dacptime']).plot(hue='output', marker='o')
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig('time_2d_eigsonly_dacponly.png')
# plt.show()

# +
# da_mean.sel(dimensions=2).sel(output=['dacpmem']).plot(hue='output', marker='o')
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig('mem_2d_eigsonly_dacponly.png')
# plt.show()
# -


