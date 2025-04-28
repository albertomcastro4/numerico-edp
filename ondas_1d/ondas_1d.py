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

# Obtención de la matriz (sparse.csr_matrix) para n arbitrario.
def method_matrix(n: int) -> sparse.csr_matrix:
    diagonals = [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)]
    offsets = [0, -1, 1]
    A = sparse.diags(diagonals, offsets, format='csr')
    return A

# Método de diferencias finitas para resolver el problema de contorno.
def method(n: int, m: int, init_cond: function_t, init_cond_der: function_t, a: float = 0, b: float = 1, t_0: float = 0, t_1: float = -1, 
        plotter: Callable[[np.ndarray, np.ndarray], None] = lambda x : -1       
    ) -> list[np.ndarray]:
    if t_1 < t_0: t_1 = t_0 + m
    x = np.linspace(a, b, n + 1)
    xx = x[1:-1]
    t = np.linspace(t_0, t_1, m + 1)

    u0 = np.array([init_cond(i) for i in xx])
    v0 = np.array([init_cond_der(i) for i in xx])
    h = x[1] - x[0]
    k = 1.0 / m

    L = (k**2 / h**2) * method_matrix(n - 1)
    u = u0
    old_u = u0 - k * v0
    uu = [np.concatenate(([0], u0, [0]))]
    for _ in range(1, m):
        temp_u = u
        u = 2 * u - old_u + L @ u
        u = np.squeeze(np.asarray(u))
        old_u = temp_u

        #uu.append(np.concatenate(([0], u, [0])))
        plotter(x, np.concatenate(([0], u, [0])))

    #xxx, ttt = np.meshgrid(x, tt)

    #return [xxx, ttt, np.array(uu)]

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
ax1.set_title(r'Solución de la ecuación de onda con condición inicial $u(x, 0) = u_0$')

left_v  = 0
right_v = 1
l = left_v - right_v

t_0 = 0

# Definición de las funciones u_0.
u_0: function_t = lambda x : np.sin(2 * np.pi * x / l)  #np.exp(-50*(x-0.5)**2)
v_0: function_t = lambda x : 0  #-100 * ( x - 0.5) * (np.exp(-50*(x-0.5)**2))

iters = [(50, 500)]
cputime, error, cputime_old, error_old, n_old = 0, 0, 0, 0, -1

def plotter(x, plot):
    ax1.clear()
    ax1.plot(x, plot, color='blue')
    ax1.set_ylim(-1.2, 1.2)
    plt.pause(0.1)

for i, (n, m) in enumerate(iters):
    method(n, m, u_0, v_0 ,left_v, right_v, t_0, 1,plotter = plotter)
        


ax1.legend()
plt.show()