"""
Resolver u''(x) = 1 - x, x ∈ (0, 3),
u(0) = u(3) = 0.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation as anim
from scipy import integrate, sparse
from scipy.sparse import linalg
import colorsys
from typing import Callable
import time


constant_t    = float | np.floating
function_t  = Callable[[constant_t], constant_t]
function2_t = Callable[[constant_t, constant_t], constant_t]

# Definición de las funciones u_0.
u_0: function_t = lambda x, y : np.sin(x**2+y**2)*(x-1)*(y-1)*(x+1)*(y+1)

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
def method(n: int, m: int, init_cond: function2_t, a: float = 0, b: float = 1, t_0: float = 0, t_1: float = 1) -> list[np.ndarray]:
    x = np.linspace(a, b, n + 1)
    y = np.linspace(a, b, n + 1)
    xx = x[1:-1]
    yy = y[1:-1]

    u0 = [init_cond(i, j) for j in yy for i in xx]

    h = xx[1] - xx[0]
    k = 1 / m

    L = sparse.identity((n-1)**2) - (k / h**2) * method_matrix(n)
    u = u0
    uu = []
    for _ in range(1, m):
        u = linalg.spsolve(L, u)

        zz = np.array(u).reshape(n-1, n-1)
        zz = np.concatenate((np.reshape(np.zeros((n-1)), (1, n-1)), zz, np.reshape(np.zeros((n-1)), (1, n-1))), axis=0)
        zz = np.concatenate((np.reshape(np.zeros((n+1)), (n+1, 1)), zz, np.reshape(np.zeros((n+1)), (n+1, 1))), axis=1)
        uu.append(zz)

    mesh_xx, mesh_yy = np.meshgrid(x, y)
    return [mesh_xx, mesh_yy, uu]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')
ax1.set_zlabel(r'$u(x, t)$')
ax1.set_title(r'Solución de la ecuación del calor con condición inicial $u(x, 0) = u_0$')

left_v  = -np.pi/2
right_v = np.pi/2

iters = [(50, 100)]
cputime, error, cputime_old, error_old, n_old = 0, 0, 0, 0, -1
z
for i, (n, m) in enumerate(iters):
    x, y, z = method(n, m, u_0, left_v, right_v, 0, 1)
    max_z = np.max(z)
    min_z = np.min(z)
    while True:
        for plot in z:
            ax1.clear()
            ax1.plot_surface(x, y, plot, cmap=cm.viridis, alpha=0.5)
            ax1.set_zlim(min_z, max_z)
            plt.pause(0.1)


ax1.legend()
plt.show()