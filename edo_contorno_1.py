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

f: Callable[[np.floating], float | np.floating] = lambda x : 1 - x
sol: Callable[[np.floating], np.floating] = lambda x : integrate.quad(lambda t: f(t)*(x-t), 0, x)[0] - (x / 3.0) * integrate.quad(lambda t : f(t)*(3-t), 0, 3)[0]

def method_matrix(n: int) -> sparse.csr_matrix:
    diagonals = [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)]
    offsets = [0, -1, 1]
    A = sparse.diags(diagonals, offsets, format='csr')
    return A

def method(n: int, f: Callable[[np.floating], float | np.floating], a: float, b: float) -> list[np.ndarray]:
    x = np.linspace(a, b, n + 2)
    xx = x[1:-1]

    y = [f(i) for i in xx]
    h = xx[1] - xx[0]

    L = (1/h)**2 * method_matrix(n)
    
    fn = linalg.spsolve(L, y)
    fn = np.insert(fn, 0, 0)
    fn = np.insert(fn, n + 1, 0)

    return [x, fn]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

fig, axes = plt.subplot_mosaic([['solution', 'solution'], ['solution', 'solution'], ['error,', 'time']])
fig.suptitle(r" Aproximación de la solución de $u''(x) = 1 - x, x \in (0, 3)$, $u(0) = u(3) = 0$")
ax1 = fig.axes[0]
ax1.title.set_text(r'Aproximación $u_n$ vs. solución $u(x) = \frac{1}{6}x(x-3)^2$')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$u(x)$')

ax2 = fig.axes[1]
ax2.title.set_text('Error')
ax2.set_xlabel(r'$h = \frac{1}{n}$')
ax2.set_ylabel(r'$\max \|u(x) - u_n(x)\|_{\infty}$')

ax3 = fig.axes[2]
ax3.title.set_text('Tiempo de CPU')
ax3.set_xlabel(r'$n$')
ax3.set_ylabel(r'$t (s)$')

# draw axes
for ax in axes.values():
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.grid(True, which='both', linestyle='--', lw=0.5)


left_v  = 0
right_v = 3

iters = [5, 10, 20, 50, 100, 500, 1000, 4000, 8000, 12000]
for i, n in enumerate(iters):
    start = time.perf_counter()
    x, y = method(n, f, left_v, right_v)
    cputime = time.perf_counter() - start

    color1 = colorsys.hsv_to_rgb(35 / 360.0, ((i + 1) / (len(iters)+1)) ** (0.5) , 1)
    ax1.plot(x, y, label=rf'$u_{{{n}}}$', color = color1)

    error = np.max(np.abs(y - [sol(i) for i in x]))
    ax2.plot(1.0 / n, error, 'x-', color = 'blue')
    ax3.plot(n, cputime, 'x-', color = 'red')

x = np.linspace(left_v, right_v, 100)
y = [sol(i) for i in x]
color1 = colorsys.hsv_to_rgb(35 / 360.0, 1 , 1)
ax1.plot(x, y, label='solución', color = color1)

ax1.legend()
plt.show()


