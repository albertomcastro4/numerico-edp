import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation as anim
from scipy import integrate, sparse, differentiate
from scipy.sparse import linalg
import colorsys
from typing import Callable
import time
from functools import wraps

constant_t    = float | np.floating
function_t  = Callable[[constant_t], constant_t]
function2_t = Callable[[constant_t, constant_t], constant_t]

x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)

alpha = lambda x : np.pow(x, 4)
beta = lambda x : 1 - np.cos(x)

u = lambda x, t : np.pow(t, 4) + 6*np.pow(t, 2)*np.pow(x, 2) + np.pow(x, 4) + t - np.sin(t)*np.cos(x) if t <= x else (
4*(np.pow(t, 3)*x + t*np.pow(x, 3)) + x -np.sin(x)*np.cos(t))


# u_t = lambda x, t : differentiate.derivative(u, t, args=(x,))

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
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)
ax1.tick_params(axis='z', labelsize=10)

xx, tt = np.meshgrid(x, t)
zz = np.array([[u(i, j) for i in x] for j in t])
# zz_t = np.array([[u_t(i, j) for i in x] for j in t])

ax1.plot_surface(xx, tt, zz, cmap=cm.plasma, edgecolor='none', alpha=0.5)
ax1.plot(x, np.zeros(len(x)), alpha(x), color='red', alpha=0.5, linewidth=3)
for j in range(0, len(t), int(len(t)/10)):
    ax1.plot(x, t[j]*np.ones(len(x)), zz[j], color='red', alpha=0.5, linewidth=3)
#    ax1.plot(x, t[j]*np.ones(len(x)), zz_t[j], color='red', alpha=0.5, linewidth=3)

plt.show()
