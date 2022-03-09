# %% Imports
import numpy as np
import matplotlib.pyplot as plt

# %% Data
x = [100, 50, 40, 15, 5]
y = [1, 2, 3, 4, 5]

# %% Using polyfit
p = np.polyfit(x, y, 2, rcond=None, full=False, w=None, cov=False)
print(p)

# %% Showing results
def f(p, t):
    return p[0] * t ** 2 + p[1] * t + p[2]

min_x = np.min(x)
max_x = np.max(x)

t = np.linspace(min_x, max_x, 100, endpoint=True)
plt.scatter(x, y, c='r')
plt.plot(t, f(p, t))

# %% Using poly1d
f = np.poly1d(p)
t = np.linspace(min_x, max_x, 100, endpoint=True)
plt.scatter(x, y, c='r')
plt.plot(t, f(t))

# %% Visualization
x = [130, 100, 70, 60, 30, 10]
y = [0.5 , 1, 2, 3, 4, 5]
min_x = np.min(x)
max_x = np.max(x)
p = np.polyfit(x, y, 1, rcond=None, full=False, w=None, cov=False)
print(p)
f = np.poly1d(p)
t = np.linspace(min_x, max_x, 100, endpoint=True)
plt.scatter(x, y, c='r')
plt.plot(t, f(t))

