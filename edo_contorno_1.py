import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg, integrate, sparse
import colorsys
from typing import Callable
import time

def method_matrix(n: int) -> np.ndarray:
    start = time.process_time()
    A = np.zeros((n, n))
    np.fill_diagonal(A, -2)
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    cputime = time.process_time() - start
    print(f"Took {cputime} seconds to create matrix of order {n}")
    return A

def method(n: int, f:Callable[[np.floating], float | np.floating], a:float, b:float) -> list[np.ndarray]:
    x = np.linspace(a, b, n + 2)
    xx = x[1:-1]

    y = [f(i) for i in xx]
    h = xx[1] - xx[0]

    L = (1/h)**2 * method_matrix(n)
    
    fn = linalg.solve(L, y)
    fn = np.insert(fn, 0, 0)
    fn = np.insert(fn, n + 1, 0)

    return [x, fn]
    #color1 = colorsys.hsv_to_rgb(35 / 360.0, 0.3 , 1)#@0.3 + 0.7 * (1 - (1.0 * diff1 / first_diff1)), 1)
    #plt.plot(x, fn, label=f'n={n}', color = color1)

fig, (ax1, ax2) = plt.subplots(1, 2)

f: Callable[[np.floating], float | np.floating] = lambda x : x ** 7
sol: Callable[[np.floating], np.floating] = lambda x : integrate.quad(lambda t: f(t)*(x-t), 0, x)[0] - x*integrate.quad(lambda t : f(t)*(3-t), 0, 3)[0] #x*(x-3)**2 / 6.0

# draw axes
ax1.axhline(0, color='black', lw=0.5)
ax1.axvline(0, color='black', lw=0.5)
ax2.axhline(0, color='black', lw=0.5)
ax2.axvline(0, color='black', lw=0.5)


left_v  = 0
right_v = 3

for i, n in enumerate([5, 10, 20, 50, 100, 500, 1000]):
    start = time.process_time()
    x, y = method(n, f, left_v, right_v)
    cputime = time.process_time() - start
    print(f"Took {cputime} seconds to solve the problem")

    color1 = colorsys.hsv_to_rgb(35 / 360.0, ((i + 1) /9.0) ** (0.5) , 1)
    ax1.plot(x, y, label=f'n={n}', color = color1)

    error = np.max(np.abs(y - [sol(i) for i in x]))
    ax2.plot(n, error, 'o', color = 'blue')


x = np.linspace(left_v, right_v, 100)
y = [sol(i) for i in x]
color1 = colorsys.hsv_to_rgb(35 / 360.0, 1, 1)
ax1.plot(x, y, label='solution', color = color1)

ax1.legend()
ax2.legend()
plt.show()


