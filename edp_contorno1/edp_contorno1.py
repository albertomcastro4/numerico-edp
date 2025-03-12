"""
Resolver u''(x) = 1 - x, x ∈ (0, 3),
u(0) = u(3) = 0.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate, sparse
from scipy.sparse import linalg
import colorsys
from typing import Callable
import time

# Definición de las funciones f y solución.
f: Callable[[np.floating, np.floating], float | np.floating] = lambda x, y : np.cos(3*x) * np.sin(3 * y)
#sol: Callable[[np.floating], np.floating] = lambda x : integrate.quad(lambda t: f(t)*(x-t), 0, x)[0] - (x / 3.0) * integrate.quad(lambda t : f(t)*(3-t), 0, 3)[0]
sol: Callable[[np.floating, np.floating], np.floating] = lambda x : x**2 / 2 - x**3 / 6

def matrix_to_vector_index(i: int, j: int, n: int) -> int:
    return i + j * n
def vector_to_matrix_index(k: int, n: int) -> tuple[int, int]:
    return (k % n, k // n)

# Obtención de la matriz (sparse.csr_matrix) para n arbitrario.
def method_matrix(n: int) -> sparse.csr_matrix:
    diagonals = [-4 * np.ones((n-1)**2), np.ones((n - 1)**2), np.ones((n - 1)**2), np.ones((n - 1)**2), np.ones((n - 1)**2)]
    for i in range(n-2, (n-1)**2, n-1):
        diagonals[-2][i] = 0
        diagonals[-1][i] = 0
    offsets = [0, -(n-1), n-1, 1, -1]
    A = sparse.diags(diagonals, offsets, format='csr')
    return A

# Método de diferencias finitas para resolver el problema de contorno.
def method(n: int, f: Callable[[np.floating], float | np.floating], a: float, b: float) -> list[np.ndarray]:
    x = np.linspace(a, b, n + 1)
    y = np.linspace(a, b, n + 1)
    xx = x[1:-1]
    yy = y[1:-1]

    z = [f(i, j) for j in yy for i in xx]
    h = xx[1] - xx[0]

    L = (1 / h)**2 * method_matrix(n)
    
    fn = linalg.spsolve(L, z)

    zz = np.array(fn).reshape(n-1, n-1)
    zz = np.concatenate((np.zeros((n-1, 1)), zz, np.zeros((n-1, 1))), axis=1)
    zz = np.concatenate((np.zeros((1, n+1)), zz, np.zeros((1, n+1))), axis=0)

    mesh_xx, mesh_yy = np.meshgrid(x, y)
    print(zz)

    return [mesh_xx, mesh_yy, zz]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

fig = plt.figure()
ax1 = plt.axes(projection='3d')
fig.suptitle(r" Aproximación de la solución de $u''(x) = 1 - x, x \in (0, 3)$, $u(0) = u(3) = 0$")
#Make 'projection=3d' to ax1
ax1.title.set_text(r'Aproximación $u_N$ vs. solución $u(x) = \frac{x^2}{2}-\frac{x^3}{6}$')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$u(x)$')

method_matrix(4)

left_v  = 0
right_v = 1

iters = [30]
cputime, error, cputime_old, error_old, n_old = 0, 0, 0, 0, -1
for i, n in enumerate(iters):
    start = time.perf_counter()
    x, y, z = method(n, f, left_v, right_v)
    cputime_old = cputime
    cputime = time.perf_counter() - start

    color1 = colorsys.hsv_to_rgb(35 / 360.0, ((i + 1) / (len(iters)+1)) ** (0.5) , 1)
    #create a mesh grid
    ax1.plot(x, y, z, label=f'$u_{n}$', color=color1, alpha=0.5)

    #error_old = error
    #error = np.max(np.abs(y - [sol(i) for i in x]))

    #if i > 0:
    #    ax2.plot([1/n_old, 1/n], [error_old, error], 'o', color = 'red', linestyle='-')
    #    ax3.plot([n_old, n], [cputime_old, cputime], 'o', color = 'blue', linestyle='-')
    n_old = n

#if len(x) < 100:
#    x = np.linspace(left_v, right_v, 100)
#y = [sol(i) for i in x]
#color1 = colorsys.hsv_to_rgb(35 / 360.0, 1 , 1)
#ax1.plot(x, y, label='solución', color = color1)

ax1.legend()
plt.show()