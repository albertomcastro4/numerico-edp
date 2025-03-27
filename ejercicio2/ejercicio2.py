#encoding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate, sparse
from scipy.sparse import linalg
from typing import Callable

constant_t    = float | np.floating
function_t    = Callable[[constant_t], constant_t]

# Creación de la matriz a partir de las diagonales.
def method_matrix(n: int, diag_main: np.ndarray, diag_lu: np.ndarray) -> sparse.csr_matrix:
    diagonals = [diag_main, diag_lu, diag_lu]
    offsets = [0, -1, 1]
    A = sparse.diags(diagonals, offsets, format='csr')
    return A

# Implementación del método.
def method(n: int, a_func: function_t , f: function_t, a: float, b: float) -> list[np.ndarray]:
    x = np.linspace(a, b, n + 1)
    xx = x[1:-1]

    y = [f(i) for i in xx]
    h = xx[1] - xx[0]

    diag_main = [-a_func(a + (i + 0.5) * h) - a_func(a + (i - 0.5) * h) for i in range(1, n)]
    diag_lu = [a_func(a + (i + 0.5) * h) for i in range(1, n - 1)]

    # (L+B), con B = -I
    L = (1 / h)**2 * method_matrix(n - 1, np.array(diag_main), np.array(diag_lu)) - sparse.identity(n - 1, format='csr')
    
    fn = linalg.spsolve(L, y)
    fn = np.insert(fn, 0, 0)
    fn = np.insert(fn, n, 0)

    return [x, fn]

# Preparación de la gráfica y parámetros visuales.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

fig, axes = plt.subplot_mosaic([['solution']])
fig.suptitle(r" Aproximación de la solución de $\left(a(x)u'(x)\right)'-u(x)=-f(x)$, $a(x)=2-x$, $f(x)=-e^{-20(x-x_0)^2}$")
ax1 = fig.axes[0]
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$u(x)$')

for ax in axes.values():
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(True, which='both', linestyle='--', lw=0.5)

# Datos del problema
left_v  = 0
right_v = 1

a:    function_t    = lambda x      : 2 - x
f_x0: function_t    = lambda x, x_0 : -np.exp(-20*(x - x_0)**2)
f_1:  function_t    = lambda x      : f_x0(x, 0.5)
f_2:  function_t    = lambda x      : f_x0(x, 0.9)


# Resolución del problema
iters = [1000]
for i, n in enumerate(iters):    
    x, y = method(n, a, f_1, left_v, right_v)
    ax1.plot(x, y, label=rf'$x_0=0,5$')
    integral_1 = integrate.simpson(y, x)

    x, y = method(n, a, f_2, left_v, right_v)
    ax1.plot(x, y, label=rf'$x_0=0,9$')
    integral_2 = integrate.simpson(y, x)

print("Los valores aproximados usando la regla de Simpson para las integrales son:")
print(f" - Para x_0 = 0.5: {integral_1}")
print(f" - Para x_0 = 0.9: {integral_2}")

ax1.legend()
plt.show()