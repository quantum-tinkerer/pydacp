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

calculation = 'eigvals_only_2d'

# +
t = 1


def make_syst(N=100, dimension=2):

    if dimension == 2:
        L = np.sqrt(N)
        lat = kwant.lattice.square(a=1, norbs=1)
        # Define 2D shape
        def shape(pos):
            (x, y) = pos
            return (-L < x < L) and (-L < y < L)

    elif dimension == 3:
        L = np.cbrt(N)
        lat = kwant.lattice.cubic(a=1, norbs=1)
        # Define 3D shape
        def shape(pos):
            (x, y, z) = pos
            return (-L < x < L) and (-L < y < L) and (-L < z < L)

    elif dimension == 1:
        L = N
        lat = kwant.lattice.chain(a=1, norbs=1)
        # Define 1D shape
        def shape(pos):
            x = pos[0]
            return -L < x < L

    syst = kwant.Builder()

    def onsite(site, seed):
        delta_mu = kwant.digest.uniform(repr(site.pos) + str(seed)) - 0.5
        # we multiply it by 20 to ensure the spectrum is uniformly distributed
        return 20 * delta_mu

    if dimension == 2:
        syst[lat.shape(shape, (0, 0))] = onsite
    elif dimension == 3:
        syst[lat.shape(shape, (0, 0, 0))] = onsite
    elif dimension == 1:
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
        eigs = dacp.dacp.eigvalsh(
            A=H,
            window=(-a, a),
            random_vectors=10,
        )
        return len(eigs)

    start_time = time.time()
    dacp_memory, k = memory_usage(dacp_benchmark, max_usage=True, retval=True)
    dacp_time = time.time() - start_time

    dacp_data = [dacp_time, dacp_memory]

    return dacp_data


# +
params = {
    'N' : [10**i for i in np.linspace(3, 5.5, 7, endpoint=True)],
    'seeds' : [1, 2, 3],
    'dimensions' : [2]
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

da.to_netcdf('./data/data_benchmark_' + calculation + '.nc')

da = xr.open_dataarray('./data/data_benchmark_' + calculation + '.nc')

da_mean=da.mean(dim='seeds')

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15
plt.rcParams['legend.fontsize'] = 15

from scipy.stats import linregress
from scipy.optimize import curve_fit


def slope(dim, output):
    return linregress(
        np.log10(da_mean.N.values),
        np.log10(da_mean.sel(output=output, dimensions=dim).values)
    )[0:2]


# +
def power_law(x, a, b):
    return a*x**b

def power_fit(dim, output):
    return curve_fit(
        power_law,
        da_mean.N.values,
        da_mean.sel(output=output, dimensions=dim).values
    )[0]


# -

time_dacp_fit = slope(dim=2, output='dacptime')
time_sparse_fit = slope(dim=2, output='sparsetime')
print(time_dacp_fit, time_sparse_fit)

time_dacp_fit = power_fit(dim=2, output='dacptime')
time_sparse_fit = power_fit(dim=2, output='sparsetime')
print(time_dacp_fit, time_sparse_fit)

mem_dacp_fit_0 = slope(dim=2, output='dacpmem')
mem_sparse_fit_0 = slope(dim=2, output='sparsemem')
print(mem_dacp_fit_0, mem_sparse_fit_0)

mem_dacp_fit = power_fit(dim=2, output='dacpmem')
mem_sparse_fit = power_fit(dim=2, output='sparsemem')
print(mem_dacp_fit, mem_sparse_fit)

from sys import getsizeof

getsizeof(np.random.rand(int(1e6)) + 1j * np.random.rand(int(1e6))) / 1e6 * 1000

(da_mean.sel(dimensions=2).sel(output=['dacptime', 'sparsetime'])/3600).plot(hue='output', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathrm{Time\ [hours]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
plt.yticks([1e-3, 1e-2, 1e-1, 1])
plt.tight_layout()
plt.savefig('time_2d_' + calculation + '.png')
plt.show()


def power_from_slope(x, a, b):
    return 10**b * x ** a


(da_mean.sel(dimensions=2).sel(output=['dacpmem', 'sparsemem'])/1000).plot(hue='output', marker='o')
plt.plot(np.array(params['N']), power_law(np.array(params['N']), *mem_dacp_fit)/1000)
plt.plot(np.array(params['N']), power_law(np.array(params['N']), *mem_sparse_fit)/1000)
# plt.plot(np.array(params['N']), power_from_slope(np.array(params['N']), *mem_sparse_fit)/1000)
# plt.plot(np.array(params['N']), power_from_slope(np.array(params['N']), *mem_dacp_fit)/1000)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$\mathrm{Memory\ [GB]}$')
plt.xlabel(r'$\mathrm{Number\ of\ sites}$')
# plt.yticks([0.1, 1, 10])
plt.tight_layout()
plt.savefig('mem_2d_' + calculation + '.png')
plt.show()

(da_mean.sel(dimensions=2).sel(output=['dacpmem'])/1000).data

params['N']

mem_sparse_fit/1000

# plt.plot(np.array(params['N']), power_law(np.array(params['N']), *mem_dacp_fit)/1000)
plt.plot(np.array(params['N']), power_law(np.array(params['N']), *mem_sparse_fit)/1000)
plt.xscale('log')
plt.yscale('log')

np.array(params['N'])

0.7*10**1.3


