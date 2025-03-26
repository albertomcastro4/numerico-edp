"""
Resolver u''(x) = 1 - x, x ∈ (0, 3),
u(0) = u(3) = 0.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import integrate, sparse
from scipy.sparse import linalg
import colorsys
from typing import Callable
import time

function_t  = Callable[[float | np.floating], float | np.floating]
function2_t = Callable[[float | np.floating, float | np.floating], float | np.floating]
constant_t    = float | np.floating
class square_boundary_condition:
    def __init__(self, left: function2_t | constant_t = 0, right: function2_t | constant_t = 0, bottom: function2_t | constant_t = 0, top: function2_t | constant_t = 0) -> None:
        self.left = left if callable(left) else lambda x, y : left
        self.right = right if callable(right) else lambda x, y : right
        self.bottom = bottom if callable(bottom) else lambda x, y : bottom
        self.top = top if callable(top) else lambda x, y : top

# Definición de las funciones f y solución.
f: function2_t = lambda x, y : 1 - np.sqrt(x**2 + y**2)

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
def method(n: int, f: function2_t, a: constant_t, b: constant_t, boundary_condition: square_boundary_condition = square_boundary_condition()) -> list[np.ndarray]:
    if b < a:
        swap = a; a = b; b = swap

    x = np.linspace(a, b, n + 1)
    y = np.linspace(a, b, n + 1)
    xx = x[1:-1]
    yy = y[1:-1]

    z = [f(i, j) for j in yy for i in xx]
    h = xx[1] - xx[0]

    L = (1 / h)**2 * method_matrix(n)
    boundary_values = {
        "left" : [0.5 * (boundary_condition.left(a, b) + boundary_condition.top(a,b))] + [boundary_condition.left(a, l) for l in yy] + [0.5 * (boundary_condition.left(a, b) + boundary_condition.bottom(a,b))],
        "right" : [0.5 * (boundary_condition.right(a, b) + boundary_condition.top(b,b))] + [boundary_condition.right(b, l) for l in yy] + [0.5 * (boundary_condition.right(a, b) + boundary_condition.bottom(b,b))],
        "bottom" : [0.5 * (boundary_condition.bottom(a, b) + boundary_condition.left(a,a))] + [boundary_condition.bottom(l, a) for l in xx] + [0.5 * (boundary_condition.bottom(a, b) + boundary_condition.right(b,a))],
        "top" : [0.5 * (boundary_condition.top(a, b) + boundary_condition.left(b,b))] + [boundary_condition.top(l, b) for l in xx] + [0.5 * (boundary_condition.top(a, b) + boundary_condition.right(b,b))]
    }
    boundary_vector = sparse.csr_matrix(np.concatenate([boundary_values["bottom"], np.zeros((n-1)**2), boundary_values["top"]]))
    zz = linalg.spsolve(L, z + sparse.flatten(boundary_vector).tolist())

    zz = np.array(zz).reshape(n-1, n-1)
    zz = np.concatenate((np.reshape(boundary_values["bottom"][1:-1], (1, n - 1)), zz, np.reshape(boundary_values["top"][1:-1], (1, n - 1))), axis=0)
    zz = np.concatenate((np.reshape(boundary_values["left"], (n + 1, 1)), zz, np.reshape(boundary_values["right"], (n + 1, 1)),), axis=1)
    mesh_xx, mesh_yy = np.meshgrid(x, y)
    return [mesh_xx, mesh_yy, zz]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

fig, axes = plt.subplot_mosaic([['main', 'contour']], per_subplot_kw= {
    'main': {"projection" : "3d"},
    'contour': {},
})

fig.suptitle(r" Aproximación de la solución de $\Delta u(x) = f, x \in ]0, 1)[\times]0,1[$, $\left.u\right|_{\partial\Omega} = 0$")
#Make 'projection=3d' to ax1
#ax1.title.set_text(r'Aproximación $u_N$ vs. solución $u(x) = \frac{x^2}{2}-\frac{x^3}{6}$')
#ax1.set_xlabel(r'$x$')
#ax1.set_ylabel(r'$u(x)$')

ax1, ax2 = axes['main'], axes['contour']

method_matrix(4)

left_v  = 0
right_v = 1

iters = [100]
cputime, error, cputime_old, error_old, n_old = 0, 0, 0, 0, -1

boundary_condition = square_boundary_condition(1, 1, 1, 1)

for i, n in enumerate(iters):
    start = time.perf_counter()
    x, y, z = method(n, f, left_v, right_v, boundary_condition)
    cputime_old = cputime
    cputime = time.perf_counter() - start

    color1 = colorsys.hsv_to_rgb(35 / 360.0, ((i + 1) / (len(iters)+1)) ** (0.5) , 1)
    ax1.plot_surface(x, y, z, label=f'$u_{n}$', cmap = cm.viridis, alpha=0.7)
    ax2.contour(x, y, z, cmap = cm.viridis, alpha=0.7)

#if len(x) < 100:
#    x = np.linspace(left_v, right_v, 100)
#y = [sol(i) for i in x]
#color1 = colorsys.hsv_to_rgb(35 / 360.0, 1 , 1)
#ax1.plot(x, y, label='solución', color = color1)

ax1.legend()
plt.show()