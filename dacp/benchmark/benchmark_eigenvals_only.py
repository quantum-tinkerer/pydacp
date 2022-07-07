import kwant
import kwant.linalg.mumps as mumps
import dacp
import time
import scipy.sparse.linalg as sla
from scipy.sparse import identity, diags
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from itertools import product
from tqdm import tqdm
import itertools as it
import xarray as xr

calculation = 'eigvals_only'

# +
t = 1

def make_syst(N=100, dimension=2):
    mu=2*dimension-1

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
        return delta_mu - mu

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
    print(
        "Starting DACP step. Iteration "
        + str(int(seed))
        + ". "
        + str(int(dimension))
        + "-dimensional system. "
        + str(int(N))
        + " atoms."
    )
    print(a)

    def dacp_benchmark():
        eigs = dacp.dacp.eigh(
            H,
            window_size=a,
            eps=0.05,
            return_eigenvectors=False,
            filter_order=15,
            error_window=0.0,
            random_vectors=5,
        )
        return len(eigs)

    start_time = time.time()
    dacp_memory, k = memory_usage(dacp_benchmark, max_usage=True, retval=True)
    dacp_time = time.time() - start_time

    dacp_data = [dacp_time, dacp_memory]
    print(
        "Starting sparse diagonalization step. Iteration "
        + str(int(seed))
        + ". "
        + str(int(dimension))
        + "-dimensional system. "
        + str(int(N))
        + " atoms."
    )

    def sparse_benchmark():
        _ = sparse_diag(H, sigma=0, k=k, which="LM", return_eigenvectors=False)

    start_time = time.time()
    sparse_memory, a = memory_usage(sparse_benchmark, max_usage=True, retval=True)
    sparse_time = time.time() - start_time

    sparse_data = [sparse_time, sparse_memory]

    return sparse_data, dacp_data


# +
params = {
    'N' : [10**i for i in np.linspace(3, 4.5, 5, endpoint=True)],
    'seeds' : [1, 2, 3],
    'dimensions' : [2, 3]
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
shapes = [*shapes, 4]
shaped_data = np.reshape(result, shapes)

shaped_data = np.reshape(result, shapes)
da = xr.DataArray(
    data=shaped_data,
    dims=[*params.keys(), 'output'],
    coords={
        **params,
        'output': ['sparsetime', 'sparsemem', 'dacptime', 'dacpmem']
    }
)

da.to_netcdf('./benchmark_data/data_benchmark_' + calculation + '.nc')

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
slope(dim=1, output='sparsetime')

slope(dim=2, output='dacptime')
slope(dim=2, output='sparsetime')

slope(dim=3, output='dacptime')
slope(dim=3, output='sparsetime')

slope(dim=1, output='dacpmem')
slope(dim=1, output='sparsemem')

slope(dim=2, output='dacpmem')
slope(dim=2, output='sparsemem')

slope(dim=3, output='dacpmem')
slope(dim=3, output='sparsemem')

from sys import getsizeof

getsizeof(np.random.rand(int(1e6)) + 1j * np.random.rand(int(1e6))) / 1e6 * 1000

(da_mean.sel(dimensions=1).sel(output=['dacptime', 'sparsetime'])/3600).plot(hue='output', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathrm{Time\ [hours]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
plt.tight_layout()
plt.savefig('time_1d_' + calculation + '.png')
plt.show()

(da_mean.sel(dimensions=1).sel(output=['dacpmem', 'sparsemem'])/1000).plot(hue='output', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathrm{Memory\ [GB]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
plt.tight_layout()
plt.savefig('mem_1d_' + calculation + '.png')
plt.show()

(da_mean.sel(dimensions=2).sel(output=['dacptime', 'sparsetime'])/3600).plot(hue='output', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathrm{Time\ [hours]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
plt.tight_layout()
plt.savefig('time_2d_' + calculation + '.png')
plt.show()

(da_mean.sel(dimensions=2).sel(output=['dacpmem', 'sparsemem'])/1000).plot(hue='output', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathrm{Memory\ [GB]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
plt.tight_layout()
plt.savefig('mem_2d_' + calculation + '.png')
plt.show()

(da_mean.sel(dimensions=3).sel(output=['dacptime', 'sparsetime'])/3600).plot(hue='output', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathrm{Time\ [hours]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
plt.tight_layout()
plt.savefig('time_3d_' + calculation + '.png')
plt.show()

(da_mean.sel(dimensions=3).sel(output=['dacpmem', 'sparsemem'])/1000).plot(hue='output', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathrm{Memory\ [GB]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
plt.tight_layout()
plt.savefig('mem_3d_' + calculation + '.png')
plt.show()







N=1000
dimension=3
seed=1
syst = make_syst(N=N, dimension=dimension)
H = syst.hamiltonian_submatrix(params=dict(seed=seed), sparse=True)
lmax = float(sla.eigsh(H, k=1, which="LA", return_eigenvectors=False))
lmin = float(sla.eigsh(H, k=1, which="SA", return_eigenvectors=False))
print(lmin, lmax)




