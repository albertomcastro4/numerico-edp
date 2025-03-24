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

# Definición de las funciones u_0.
u_0: Callable[[np.floating], float | np.floating] = lambda x : 1

# Obtención de la matriz (sparse.csr_matrix) para n arbitrario.
def method_matrix(n: int) -> sparse.csr_matrix:
    diagonals = [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)]
    offsets = [0, -1, 1]
    A = sparse.diags(diagonals, offsets, format='csr')
    return A

# Método de diferencias finitas para resolver el problema de contorno.
def method(n: int, m: int, init_cond: Callable[[np.floating], float | np.floating], a: float = 0, b: float = 1, t_0: float = 0, t_1: float = 1) -> list[np.ndarray]:
    x = np.linspace(a, b, n + 1)
    xx = x[1:-1]
    t = np.linspace(t_0, t_1, m + 1)
    tt = t[1:-1]

    u0 = [init_cond(i) for i in xx]
    h = x[1] - x[0]
    k = 1 / m

    L = method_matrix(n - 1)
    u = u0
    uu = []
    print(np.shape(L), np.shape(u))
    for _ in range(1, m):
        print(f"{_} : {((k / h**2) * L) @ u}")
        u = u + ((k / h**2) * L) @ u
        uu.append([0] + u + [0])
    
    xxx, ttt = np.meshgrid(x, t)
    print(np.shape(xxx), np.shape(ttt), np.shape(uu))
    return [xxx, ttt, np.array(uu)]

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
ax1.set_ylabel(r'$t$')

method_matrix(4)

left_v  = 0
right_v = 1

iters = [(10, 5000)]
cputime, error, cputime_old, error_old, n_old = 0, 0, 0, 0, -1
for i, (n, m) in enumerate(iters):
    start = time.perf_counter()
    x, y, z = method(n, m, u_0)
    cputime_old = cputime
    cputime = time.perf_counter() - start

    color1 = colorsys.hsv_to_rgb(35 / 360.0, ((i + 1) / (len(iters)+1)) ** (0.5) , 1)
    #create a mesh grid
    ax1.plot_surface(x, y, z, label=f'$u_{n}$', color=color1, alpha=0.5)

    #error_old = error
    #error = np.max(np.abs(y - [sol(i) for i in x]))

    #if i > 0:
    #    ax2.plot([1/n_old, 1/n], [error_old, error], 'o', color = 'red', linestyle='-')
    #    ax3.plot([n_old, n], [cputime_old, cputime], 'o', color = 'blue', linestyle='-')
    n_old = n

#if len(x) < 100:
#    x = np.linspace(left_v, right_v, 10def method_matrix(n: int) -> sparse.csr_matrix: diagonals = [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)] offsets = [0, -1, 1] A = sparse.diags(diagonals, offsets, format='csr') return A0)
#y = [sol(i) for i in x]
#color1 = colorsys.hsv_to_rgb(35 / 360.0, 1 , 1)
#ax1.plot(x, y, label='solución', color = color1)

ax1.legend()
plt.show()