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
u_0: function_t = lambda x : np.exp(-10 * (x)**2)

# Obtención de la matriz (sparse.csr_matrix) para n arbitrario.
def method_matrix(n: int) -> sparse.csr_matrix:
    diagonals = [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)]
    offsets = [0, -1, 1]
    A = sparse.diags(diagonals, offsets, format='csr')
    return A

# Método de diferencias finitas para resolver el problema de contorno.
def method(n: int, m: int, init_cond: function_t, a: float = 0, b: float = 1, t_0: float = 0, t_1: float = 1) -> list[np.ndarray]:
    x = np.linspace(a, b, n + 1)
    xx = x[1:-1]
    t = np.linspace(t_0, t_1, m + 1)
    tt = t[:-1]

    u0 = [init_cond(i) for i in xx]
    h = x[1] - x[0]
    k = 1.0 / m

    L = method_matrix(n - 1)
    L[0, 0] = -1
    L = np.identity(n-1) - (k / h**2) * L
    u = u0
    uu = [np.concatenate(([u0[0]], u0, [0]))]
    for _ in range(1, m):
        u = linalg.spsolve(L, u)
        uu.append(np.concatenate(([u[0]], u, [0])))

    xxx, ttt = np.meshgrid(x, tt)

    return [xxx, ttt, np.array(uu)]

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

left_v  = 0
right_v = 1

iters = [(100, 300)]
cputime, error, cputime_old, error_old, n_old = 0, 0, 0, 0, -1
for i, (n, m) in enumerate(iters):
    x, y, z = method(n, m, u_0, left_v, right_v, 0, 1)

    
    for j in range(0, m, int(m / 10)):
        color1 = colorsys.hsv_to_rgb(35 / 360.0, 1.0, 1.0)
        ax1.plot(x[j], y[j], z[j], color=color1, linewidth=3)
        #plt.pause(0.1)
    
    ax1.plot_surface(x, y, z, cmap=cm.plasma, alpha=0.5)


ax1.legend()
plt.show()