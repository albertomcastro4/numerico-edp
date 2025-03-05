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
f: Callable[[np.floating], float | np.floating] = lambda x : 3
#sol: Callable[[np.floating], np.floating] = lambda x : integrate.quad(lambda t: f(t)*(x-t), 0, x)[0] - (x / 3.0) * integrate.quad(lambda t : f(t)*(3-t), 0, 3)[0]
sol: Callable[[np.floating], np.floating] = lambda x : 3.0 / 2 * (x**2) + 1.0 / 2

# Obtención de la matriz (sparse.csr_matrix) para n arbitrario.
def method_matrix(n: int) -> sparse.csr_matrix:
    diagonals = [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)]
    offsets = [0, -1, 1]
    A = sparse.diags(diagonals, offsets, format='csr')
    A[0, 0] = -1
    return A

# Método de diferencias finitas para resolver el problema de contorno.
def method(n: int, f: Callable[[np.floating], float | np.floating], a: float, b: float, alpha: float = 0, beta: float = 0) -> list[np.ndarray]:
    x = np.linspace(a, b, n + 2)
    xx = x[1:-1]

    y = [f(i) for i in xx]
    h = xx[1] - xx[0]
    y[-1] -= beta / h ** 2
    
    L = (1 / h)**2 * method_matrix(n)
    
    fn = linalg.spsolve(L, y)
    fn = np.insert(fn, 0, fn[0])
    fn = np.insert(fn, len(fn), beta)

    return [x, fn]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

fig, axes = plt.subplot_mosaic([['solution', 'solution'], ['solution', 'solution'], ['error,', 'time']])
fig.suptitle(r" Aproximación de la solución de $u''(x) = 3 , x \in (0, 1)$, $u'(0) = 0$ y $u(1) = 2$")
ax1 = fig.axes[0]
ax1.title.set_text(r'Aproximación $u_N$ vs. solución $u(x) = \frac{3x^2}{2}-\frac{1}{2}$')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$u(x)$')

ax2 = fig.axes[1]
ax2.title.set_text('Error')
ax2.set_xlabel(r'$h = \frac{1}{N}$')
ax2.set_ylabel(r'$\|u - u_N\|_{\infty}$')

ax3 = fig.axes[2]
ax3.title.set_text('Tiempo de CPU')
ax3.set_xlabel(r'$N$')
ax3.set_ylabel(r'$t (s)$')

# draw axes
for ax in axes.values():
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(True, which='both', linestyle='--', lw=0.5)

left_v  = 0
right_v = 1

iters = [5, 10, 20, 50, 100, 500, 1000, 10000, 20000]
cputime, error, cputime_old, error_old, n_old = 0, 0, 0, 0, -1
for i, n in enumerate(iters):
    start = time.perf_counter()
    x, y = method(n, f, left_v, right_v, 0, 2)
    cputime_old = cputime
    cputime = time.perf_counter() - start

    color1 = colorsys.hsv_to_rgb(35 / 360.0, ((i + 1) / (len(iters)+1)) ** (0.5) , 1)
    ax1.plot(x, y, label=rf'$u_{{{n}}}$', color = color1)

    error_old = error
    error = np.max(np.abs(y - [sol(i) for i in x]))

    if i > 0:
        ax2.plot([1/n_old, 1/n], [error_old, error], 'o', color = 'red', linestyle='-')
        ax3.plot([n_old, n], [cputime_old, cputime], 'o', color = 'blue', linestyle='-')
    n_old = n

if len(x) < 100:
    x = np.linspace(left_v, right_v, 100)
y = [sol(i) for i in x]
color1 = colorsys.hsv_to_rgb(35 / 360.0, 1 , 1)
ax1.plot(x, y, label='solución', color = color1)

ax1.legend()
plt.show()