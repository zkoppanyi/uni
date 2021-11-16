# %% External imports
%matplotlib qt

import numpy as np
import matplotlib.pyplot as plt

# %% Internal imports
import viz

# %% Inputs
R =[[0.46224950999999997,  -0.88642706999999998,  0.023925829999999999],
    [-0.88672448999999998, -0.46227224, 0.0049042499999999998],
    [0.0067129900000000003, -0.023482610000000001,  -0.99970170999999997]]
t = [37.288039423382166, -10.165071594832114, 1.8727184629264104]
P = [43.997501373291016, -21.606561660766602, -78.306999206542969]
n = [2.7196987502975389e-05, 0.0013762940652668476, -0.99999904632568359]

R = np.array(R)
t = np.array(t)
P = np.array(P)
n = np.array(n)

X = -R.transpose() @ t

# %% Compute intersection
pn_0 = np.array([0, 0, 1])
pn = R @ pn_0
pp = X
rp = P
rv = n

diff = rp - pp
prod1 = diff.dot(pn)
prod2 = rv.dot(pn)
d = prod1 / prod2
ip = rp - rv * d


# %% Plotting in world
fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
viz.plot_fustrum(ax, X, R, f=1, scale=10)

ax.plot3D(P[0], P[1], P[2], c='g', marker='o')
s = 100
ax.plot3D([P[0], P[0] + n[0]*s], [P[1], P[1] + n[1]*s], [P[2], P[2] + n[2]*s], c='g', marker='o')
ax.plot3D(ip[0], ip[1], ip[2], c='r', marker='o')
viz.set_3d_axes_equal(ax)

# %% Compute angle between camera direction and normal
v = X - P
v = v / np.linalg.norm(v)
cosalpha = v.dot(n) / (np.linalg.norm(v) * np.linalg.norm(n))
alpha = np.arccos(cosalpha)/np.pi*180
print(f'{alpha} deg')

# %%
fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')

s = 100
ax.plot3D([P[0], P[0] + n[0]*s], [P[1], P[1] + n[1]*s], [P[2], P[2] + n[2]*s], c='g', marker='o')
ax.plot3D([P[0], P[0] + v[0]*s], [P[1], P[1] + v[1]*s], [P[2], P[2] + v[2]*s], c='r', marker='o')
viz.set_3d_axes_equal(ax)

# %% Plotting in camera coordinate system
# ip_cam = R.transpose() @ (ip - X)
# # ip_cam[2] ~= 0 must be
# print(ip_cam)

# fig = plt.figure(figsize=(12,10))
# ax = plt.axes(projection='3d')
# viz.plot_fustrum(ax, np.array([0, 0, 0]), np.eye(3), f=1, scale=1)
# ax.plot3D(ip_cam[0], ip_cam[1], ip_cam[2], c='r', marker='o')
# viz.set_3d_axes_equal(ax)

# %%
