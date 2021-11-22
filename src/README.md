## pyDACP: a python package for the Dual applications of Chebyshev polynomials method

### Why to even care?

Do you work with: \* Systems with large Hamiltonains? \* Interested only
within a small energy window?

Then DACP is perfect for you: the idea is to span the low-energy
subspace with two applications of Chebyshev polynomials. So if you
original Hamiltonian had *n* states and within the small window there
are only *m* states, you only have to diagonalize an *m* × *m* matrix.

### First application of Chebyshev polynomials

We write an arbitrary vector in terms of the eigenvectors of the
Hamiltonian ℋ:

The idea now is to obtain an energy-filtered vector \|*r*<sub>*E*</sub>⟩
by removing the second term of the equation above. To do so, we define
the operator with
*E*<sub>*c*</sub> = *E*<sub>*m**a**x*</sub><sup>2</sup> + *a*<sup>2</sup>,
and
*E*<sub>0</sub> = *E*<sub>*m**a**x*</sub><sup>2</sup> − *a*<sup>2</sup>.

For a large enough *k*, the *k*-th order Chebyshev polynomial of ℱ is
which indeed filters states within the \[−*a*,*a*\] window. So,

Let’s see an example!

``` python
from pyDACP import core, chebyshev
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.linalg import eigh, eig, eigvalsh
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
```

``` python
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)
plt.rcParams["figure.figsize"] = (4, 3)
plt.rcParams["lines.linewidth"] = 0.65
plt.rcParams["font.size"] = 16
plt.rcParams["legend.fontsize"] = 16
```

``` python
N = 1000
np.random.seed(1)
c = 2 * (np.random.rand(N-1) + np.random.rand(N-1)*1j - 0.5 * (1 + 1j))
b = 2 * (np.random.rand(N) - 0.5)

H = diags(c, offsets=-1) + diags(b, offsets=0) + diags(c.conj(), offsets=1)

plt.matshow(H.toarray().real, cmap='bwr', vmin=-1, vmax=1)
plt.show()
```

``` python
%%time
dacp=core.DACP_reduction(H, a=0.2, eps=0.05)
```

``` python
%%time
true_eigvals, true_eigvecs = eig(H.todense())
v_proj = dacp.get_filtered_vector()
```

``` python
plt.scatter(np.real(true_eigvals), np.log(np.abs(true_eigvecs.T.conj()@v_proj)), c='k')
plt.xlim(-3*dacp.a, 3*dacp.a)
plt.axvline(-dacp.a, ls='--', c='k')
plt.axvline(dacp.a, ls='--', c='k')
plt.ylabel(r'$|c|$')
plt.xlabel(r'$E$')
plt.show()
```

### Second application of Chebyshev polynomials: the Chebyshev evolution

Now we have one single vector \|*r*<sub>*E*</sub>⟩ within the energy
window we want. And we use again Chebyshev polynomials to span the full
basis of the subspace 𝕃 with *E* ∈ \[−*a*,*a*\]. For that, we define a
second operator, 𝒢, which is simply the rescaled Hamiltonian such that
all eigenvalues are within \[−1,1\]: with
*E*′<sub>*c*</sub> = (*E*<sub>*m**a**x*</sub>+*E*<sub>*m**i**n*</sub>)/2,
and
*E*′<sub>0</sub> = (*E*<sub>*m**a**x*</sub>−*E*<sub>*m**i**n*</sub>)/2.

A full basis is then simply: with *X* := *π*𝒢/*a*<sub>*r*</sub>, and
*a*<sub>*r*</sub> = *a*/max (\|*E*<sub>*m**a**x*</sub>\|,\|*E*<sub>*m**i**n*</sub>\|).

In fact, we can span the basis above by, instead of computing sin  and
cos  of a matrix, computing simply several Chebyshev **polynomials** of
𝒢.

The remaininig problem is that we don’t know the value of *n*, so we
must (over)estimate the dimension of this subspace. And guess what: we
use **again** Chebyshev polynomials by performing a low-resolution KPM.
Since we overestimate the dimension, we also want to get rid of linearly
dependent vectors, so we do SVD.

The final set of vectors {*ψ*<sub>*k*</sub>} is then used to compute the
projected low-energy Hamiltonian:

Let’s see how good this Hamiltonian is:

``` python
%%time
ham_red=dacp.get_subspace_matrix()
```

``` python
plt.matshow(ham_red.real, vmin=-.1, vmax=.1, cmap='bwr')
plt.show()
```

``` python
%%time
red_eigvals, red_eigvecs = eig(ham_red)
```

``` python
# This part is quite unstable.
# Half of the bandwidth
half_bandwidth=np.max(np.abs(true_eigvals))
# Number 50 times smaller than level spacing.
err=half_bandwidth/N/2
# Get indices
indx1=np.min(red_eigvals)-err<=true_eigvals
indx2=true_eigvals<=np.max(red_eigvals)+err
indx=indx1 * indx2
# Extract real eigenvalues within the desired window.
window_eigvals=true_eigvals[indx]
```

``` python
plt.plot(np.sort(window_eigvals), np.sort(red_eigvals), '-o', c='k')
plt.xlabel(r'$E_n^{big}$')
plt.ylabel(r'$E_n^{small}$')
plt.xlim(-dacp.a, dacp.a)
plt.ylim(-dacp.a, dacp.a)
plt.show()
```

``` python
# res=np.sort(red_eigvals).copy()[:window_eigvals.shape[0]]
# res-=np.sort(window_eigvals)
res=np.sort(red_eigvals)-np.sort(window_eigvals)
plt.plot(np.sort(window_eigvals), np.log(res/dacp.a), '-o', c='k')
plt.ylabel(r'$\log(E_n^{big} - E_n^{small})$')
plt.xlabel(r'$E_n^{big}$')
plt.xlim(-dacp.a, dacp.a)
plt.show()
```

## The degeneracy problem and next steps

This method works fine if there are no degeneracies. For degenerate
states, however, we get a single eigenstate. The way to work around this
problem is to first use as many random vectors as there are degenerate
states. Since random vectors are in general linearly independent, they
should lead to different degenerate states.

However, “in general” is not as formal as we wanted it to be. So ideally
we want to make sure the random vectors are actually orthogonal. And it
turns out that this is painful.

But let’s see how it goes anyway.

``` python
import kwant

a, t = 1, 1
mu=0
L = 15

lat = kwant.lattice.honeycomb(a, norbs=1)

syst = kwant.Builder()

# Define the quantum dot
def circle(pos):
    (x, y) = pos
    return (-L < x < L) and (-L < y < L)

syst[lat.shape(circle, (0, 0))] = - mu
# hoppings in x-direction
syst[lat.neighbors()] = -t
# hoppings in y-directions
syst[lat.neighbors()] = -t

fsyst=syst.finalized()

kwant.plot(syst)
plt.show()
```

``` python
%%time
H=fsyst.hamiltonian_submatrix(sparse=True)
dacp=core.DACP_reduction(H, a=0.2, eps=0.05, random_vectors=15)
```

``` python
%%time
ham_red=dacp.get_subspace_matrix()
```

``` python
evals = eigvalsh(ham_red)
n=np.arange(-evals.shape[0]/2, evals.shape[0]/2)
true_vals = eigvalsh(H.todense())
plt.scatter(n, evals, c='k')
n_true=np.arange(-true_vals.shape[0]/2, true_vals.shape[0]/2)
plt.scatter(n_true, true_vals, c='r', s=4)
plt.xlim(-20, 20)
plt.ylim(-dacp.a, dacp.a)
plt.xlabel(r'$n$')
plt.ylabel(r'$E - E_F$')
plt.show()
```

### Gram-Schmidt othogonalization

A simple way to perform this orthogonalization is via Gram-Schmidt
method. Say we already collected *m* vectors after the first Chebyshev
evolution, and let’s assume all of them are orthogonal. And then we
generate a second random vector \|*r*<sub>2</sub>⟩. Then we
orthogonalize it by simply computing:

This method is, however, unstable: the error is too large.

### QR decomposition with Householder reflections

A stable method is orthogonalization via Householder reflections.

#### Householder reflections

Say you have an arbitrary *n*-dimensional vector \|*x*<sub>1</sub>⟩, and
you want to make a reflection via an operator *H*<sub>1</sub> on it such
that *H*<sub>1</sub>\|*x*<sub>1</sub>⟩ = *e*<sub>1</sub>. Thus, you only
have to find the vector \|*v*⟩ perpendicular to plane for this
reflection, and then:
*H*<sub>1</sub> = *I* − \|*v*⟩⟨*v*\|.

#### QR decomposition

The QR decomposition is a decomposition of a matrix *A* as *A* = *Q**R*,
where *Q* is an unitary matrix and *R* is an upper triangular matrix. We
can actually go from *A* to *R* by performing a sucession of Householder
reflections: \* Write *A* as a sequence of vectors:
*A* = \[\|*x*<sub>1</sub>⟩,\|*x*<sub>2</sub>⟩,⋯,\|*x*<sub>*m*</sub>⟩\].
\* Perform a Householder reflection for \|*x*<sub>1</sub>⟩, so
*H*<sub>1</sub>*A* = \[*e*<sub>1</sub>,*H*<sub>1</sub>\|*x*<sub>2</sub>⟩,⋯,*H*<sub>1</sub>\|*x*<sub>*m*</sub>⟩\].
\* Perform a Householder reflection *H̃*<sub>2</sub> in the last *n* − 1
components of \|*x*<sub>2</sub>⟩, such that:
$$
H_2 H_1 A = \\left\[ \\begin{array}{cccc}
1 & H_1\|x_2\\rangle & \\ & \\ \\\\
0 & 1 & \\cdots & H_2 H_1\|x_m\\rangle \\\\
0 & 0 & \\ & \\ \\\\
\\vdots & \\vdots & \\ddots & \\vdots 
\\end{array} \\right\]
$$
\* If all the vectors are linearly independent, we end up with a upper
triangular matrix after performing these rotations *m* times. Therefore,
*A* = *Q**R* = (*H*<sub>*m*</sub>⋯*H*<sub>1</sub>)<sup>†</sup>*R*.

Since *Q* is unitary, we get an orthogonal basis.

#### Our workflow

At each step of the Chebyshev evolution, we get a vector
\|*ψ*<sub>*k*</sub>⟩, which we Household reflect and generate
*Q*<sup>†</sup>. Then we go to the next evolution step, and compute
*Q*<sup>†</sup>\|*ψ*<sub>*k* + 1</sub>⟩. If all the last *m* − (*k*+1)
components of this vector are zero, we know this vector is linearly
dependent to the previous ones. Then we stop the evolution. Otherwise,
we go to the next step.

When we stop the evolution (at step *m*), we collect
`Q.T.conj()[:, :m]`, which is the current basis. Then ensure the next
random vector is within the orthogonal complement of this basis and keep
going until there are no more degeneracies.

#### How to know there are no more degeneracies?

-   Start with 2 perpendicular random vectors.
-   See whether they lead to degenerate states.
-   If so, add one more vector, and check if there are 3-fold
    degeneracies.
-   Go on until at step *n* the filtered vector in the orthogonal
    complement of the current basis is the null vector.
