import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg, integrate
import colorsys
from typing import Callable
import time, math

def method_matrix(n: int) -> np.ndarray:
    start = time.process_time()
    A = np.zeros((n, n))
    np.fill_diagonal(A, -2)
    np.fill_diagonal(A[1:], 1)
    np.fill_diagonal(A[:, 1:], 1)
    cputime = time.process_time() - start
    print(f"Took {cputime} seconds to create matrix of order {n}")
    return A

def method(n: int):
    x = np.linspace(0, 1, n + 2)
    #erase first and last element
    xx = x[1:-1]

    y = [f(i) for i in xx]
    h = xx[1] - xx[0]

    L = (1/h)**2 * method_matrix(n)
    
    fn = linalg.solve(L, y)
    fn = np.insert(fn, 0, 0)
    fn = np.insert(fn, n + 1, 0)

    color1 = colorsys.hsv_to_rgb(35 / 360.0, 0.3 , 1)#@0.3 + 0.7 * (1 - (1.0 * diff1 / first_diff1)), 1)
    plt.plot(x, fn, label=f'n={n}', color = color1)

f: Callable[[np.floating], float | np.floating] = lambda x : x
sol: Callable[[np.floating], np.floating] = lambda x : integrate.quad(lambda t: f(t)*(x-t), 0, x)[0] - x*integrate.quad(lambda t : f(t)*(1-t), 0, 1)[0]

plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)

start = time.process_time()
method(10000)
cputime = time.process_time() - start
print(f"Took {cputime} seconds to solve the problem")


x = np.linspace(0, 1, 100)
y = [sol(i) for i in x]
color1 = colorsys.hsv_to_rgb(35 / 360.0, 1, 1)
plt.plot(x, y, label='solution', color = color1)

plt.legend()
plt.show()


