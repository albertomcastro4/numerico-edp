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
u_0: function_t = lambda x : np.exp(x)
f  : function_t = lambda t : 1

# Obtención de la matriz (sparse.csr_matrix) para n arbitrario.
def method_matrix(n: int) -> sparse.csr_matrix:
    diagonals = [-1 * np.ones(n), np.ones(n - 1)]
    offsets = [0, -1]
    A = sparse.diags(diagonals, offsets, format='csr')
    return A

# Método de diferencias finitas para resolver el problema de contorno.
def method(n: int, m: int, init_cond: function_t, boundary_f : function_t, a: float = 0, b: float = 1, t_0: float = 0, t_1: float = 1) -> list[np.ndarray]:
    x = np.linspace(a, b, n + 1)
    xx = x[1:]
    t = np.linspace(t_0, t_1, m + 1)

    u0 = [init_cond(i) for i in xx]
    h = x[1] - x[0]
    k = 1 / m

    L = (k / h) * method_matrix(n)
    u = u0
    uu = [np.concatenate(([boundary_f(t_0)], u0))]
    for j in range(1, m):
        u = u + L @ u
        uu.append(np.concatenate(([boundary_f(t[j])], u)))

    return [x, np.array(uu)]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

fig = plt.figure()
ax1 = plt.axes()
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')
ax1.set_title(r'Solución de la ecuación del calor con condición inicial $u(x, 0) = u_0$')

left_v  = 0
right_v = 1

t_0 = 0
t_1 = 1

iters = [(50, 50)]
cputime, error, cputime_old, error_old, n_old = 0, 0, 0, 0, -1
for i, (n, m) in enumerate(iters):
    x, plots = method(n, m, u_0, f ,left_v, right_v, t_0, t_1)

    for i, plot in enumerate(plots):
        t = t_0 + i * 1.0 * (t_1 - t_0) / m
        ax1.clear()
        ax1.plot(x, plot, color="blue", label=f"t={t}")
        plt.pause(0.1)

plt.show()