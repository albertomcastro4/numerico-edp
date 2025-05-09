"""
Resolución de la ecuación de difusión en [0, 1] con condiciones de frontera de Neumann
en el extremo izquierdo y Dirichlet en el extremo derecho.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg
from typing import Callable


constant_t    = float | np.floating
function_t  = Callable[[constant_t], constant_t]
function2_t = Callable[[constant_t, constant_t], constant_t]

# -----------------------------------------------------------------------------
# DEFINICIÓN DEL MÉTODO DE TIPO CRANK-NICHOLSON
# -----------------------------------------------------------------------------

def method_matrix(n: int) -> sparse.csr_matrix:
    diagonals = [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)]
    offsets = [0, -1, 1]
    A = sparse.diags(diagonals, offsets, format='csr')
    A[0, 0] = -1
    return A

def method_neumann_left(n: int, m: int, init_cond: function_t, a: float = 0, b: float = 1, t_0: float = 0, t_1: float = -1, 
        plotter_func: Callable[[np.ndarray, np.ndarray], None] = lambda x : None       
    ) -> None:
    if t_1 < t_0: t_1 = t_0 + m
    
    h = (b - a) / n
    k = (t_1 - t_0) / m

    x = np.linspace(a, b, n + 1)
    xx = x[1:-1]

    u0 = [init_cond(i) for i in xx]

    g = np.zeros(n - 1)
    g[0] = (k /  h)

    I = sparse.identity(n - 1)
    L = (k / 2) * (1.0 / h ** 2) * method_matrix(n - 1)
    A = (I - L)
    B = (I + L)
    
    u = u0
    plotter_func(x, np.concatenate(([u_0(x[0])], u, [u_0(x[-1])])), time = t_0)
    for iter in range(1, m + 1):
        rhs = (B @ u) + g
        u = linalg.spsolve(A, rhs)
        plotter_func(x, np.concatenate(([u[0] + h], u, [0])), time = t_0 + iter * k)

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA GRÁFICA
# -----------------------------------------------------------------------------


fig, axes = plt.subplot_mosaic([['solution', 'solution'], ['solution', 'solution'], ['error', 'error']])
ax1 = fig.axes[0]
ax2 = fig.axes[1]

ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$t$')
ax1.set_title(r'Solución de la ecuación')

ax2.set_xlabel(r'Tiempo')
ax2.set_ylabel(r'$\ell^2$ distance')
ax2.set_title(r'Distancia')

ax2.axhline(0, color='black', lw=0.5)
ax2.axvline(0, color='black', lw=0.5)
ax2.grid(True, which='both', linestyle='--', lw=0.5)

# -----------------------------------------------------------------------------
# DATOS DEL PROBLEMA
# -----------------------------------------------------------------------------

left_v  = 0
right_v = 1

t_0 = 0
t_1 = 5

u_0: function_t = lambda x : np.exp(-20 * (x - 0.5) ** 2)
u_s: function_t = lambda x : -x + 1

# -----------------------------------------------------------------------------
# DEFINICIÓN DE LA FUNCIÓN QUE DIBUJA LA SOLUCIÓN
# -----------------------------------------------------------------------------

ax2.set_xlim(t_0, t_1)

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(error = 0, error_old = 0, time_old = 0, y_lim_min = 0, y_lim_max = 1)
def plotter(x, u, time = t_0):
    ax1.clear()
    
    ax1.axhline(0, color='black', lw=0.5)
    ax1.axvline(0, color='black', lw=0.5)
    ax1.grid(True, which='both', linestyle='--', lw=0.5)

    plotter.y_lim_min = min(min(u) - 0.1, plotter.y_lim_min)
    plotter.y_lim_max = max(max(u) + 0.1, plotter.y_lim_max)
    ax1.set_ylim(plotter.y_lim_min, plotter.y_lim_max)
    

    ax1.plot(x, u_s(x), color='red', label=r'$u_s$', linestyle='--')

    ax1.plot(x, u, color='blue', label=r'$u_{aprox}$')
    ax1.set_title(r'Solución para t = ' + str(round(time, 3)))
    ax1.legend()

    plotter.error_old = plotter.error
    plotter.error = np.sqrt(np.sum((u - u_s(x))**2) / len(x))
    if time > t_0:
        ax2.plot([plotter.time_old, time], [plotter.error_old, plotter.error], color = 'red', linestyle='-')
    plotter.time_old = time

    plt.pause(0.02)

# -----------------------------------------------------------------------------
# EJECUCIÓN DEL MÉTODO
# -----------------------------------------------------------------------------

n = 100
m = 200

method_neumann_left(n, m, u_0, left_v, right_v, t_0, t_1, plotter_func = plotter)

plt.show()