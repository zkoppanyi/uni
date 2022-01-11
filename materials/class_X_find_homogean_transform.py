# %%
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

sys.path.append("../")
from viz.viz import plot_fustrum, plot_crs, set_3d_axes_equal

# %% Source
# See: http://nghiaho.com/?page_id=671
# Derivation: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

# %% Point sets
pts_model = np.array([[541962.429, -4958327.869,3961855.719],
            [541962.941,-4958327.173,3961856.517],
            [541969.331,-4958329.38,3961852.677],
            [541956.507,-4958361.554,3961821.221],
            [541946.713,-4958316.549,3961872.07]])

pts_target = np.array([[541963.455,-4958327.442,3961856.053],
            [541963.769,-4958326.934,3961856.765],
            [541970.39,-4958328.928,3961853.079],
            [541957.3,-4958361.223,3961821.027],
            [541947.238,-4958315.964,3961872.756]])

# %% Display point sets
plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
ax.scatter3D(pts_model[:,0], pts_model[:,1], pts_model[:,2], c='r', s=50)
ax.scatter3D(pts_target[:,0], pts_target[:,1], pts_target[:,2], c='b', s=50)
set_3d_axes_equal(ax)

# %% Compute coordinates in COG
cog_model = np.mean(pts_model, axis=0)
cog_target = np.mean(pts_target, axis=0)

pts_model_cog = pts_model - cog_model
pts_target_cog = pts_target - cog_target

# %% Show coordinates in COG
plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
ax.scatter3D(pts_model_cog[:,0], pts_model_cog[:,1], pts_model_cog[:,2], c='r', s=50)
ax.scatter3D(pts_target_cog[:,0], pts_target_cog[:,1], pts_target_cog[:,2], c='b', s=50)
set_3d_axes_equal(ax)

# %% Compute coveriances
C = pts_model_cog.T @ pts_target_cog

# %% Check C matrix
print('C is a square matrix with dimension 3: ', C.shape[0], 'x', C.shape[1])

# if there is one zero eigen value than the points are lie on a 2D plane
print('C matrix is positive semi-definite: ', np.linalg.eig(C)[0])

#print('Determinant: ', np.linalg.det(C))

# %% SVD decomposition
[U, S, V] = np.linalg.svd(C)
V = V.T # mostly given in this format
C_chk = U @ np.diag(S) @ V.T
print('Check decomposition: ', np.linalg.norm(C_chk - C))

# %% Uniqueness of SVD#
# In general, the SVD is unique up to arbitrary unitary transformations applied uniformly
# to the column vectors of both U and V spanning the subspaces of each singular value, and
# up to arbitrary unitary transformations on vectors of U and V spanning the kernel and
# cokernel, respectively, of M. (Source: wiki)
# So this is also a solution:
V[:, 1] = -V[:, 1]
U [:, 1] = -U[:, 1]
C_chk = U @ np.diag(S) @ V.T
print('Check decomposition: ', np.linalg.norm(C_chk - C))

# %%
R = V @ U
t = cog_target - R @ cog_model

# %% Apply transformation
pts_model_hat = (R @ pts_model.T).T + t

# %%
plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
ax.scatter3D(pts_model[:,0], pts_model[:,1], pts_model[:,2], c='r', s=50)
ax.scatter3D(pts_target[:,0], pts_target[:,1], pts_target[:,2], c='b', s=50)
ax.scatter3D(pts_model_hat[:,0], pts_model_hat[:,1], pts_model_hat[:,2], c='g', s=50)
set_3d_axes_equal(ax)


