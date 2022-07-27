# %% Imports
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from utils.viz.viz import set_3d_axes_equal

# %% Source
# See: http://nghiaho.com/?page_id=671
# Derivation: https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

# %% Point sets

pts_model = np.array([[541962.336,-4958327.264,3961855.19],
        [541962.864,-4958326.491,3961856.004],
        [541969.244,-4958328.729,3961852.225],
        [541967.304,-4958336.117,3961838.934],
        [541942.45,-4958313.896,3961874.57]])

pts_target = np.array([[541963.551,-4958326.683,3961855.34],
        [541963.792,-4958326.153,3961856.142],
        [541970.5,-4958328.18,3961852.293],
        [541968.456,-4958335.563,3961839.102],
        [541943.247,-4958313.517,3961874.94]])

np.linalg.norm(pts_model - pts_target)


# %%

pts_model = np.array([[542469.0839485932,127503.1712572407,14.289997015287282],
        [542469.6870645597,127504.254237446,14.243031427058297],
        [542475.8111025846,127499.51301038824,14.161880714446283],
        [542473.1585485833,127484.66392089333,11.43252228931343],
        [542450.6378321772,127527.85755548254,14.32731174323642]])

pts_target = np.array([[542470.3527275253,127503.573374236,14.03570734019333],
        [542470.6449000955,127504.51395174209,14.145552052541893],
        [542477.1176194218,127499.82863801438,13.884704144826738],
        [542474.3617623677,127485.06726074591,11.205089686071032],
        [542451.4687627961,127528.33212887496,14.331774375970408]])

# np.linalg.norm(pts_model - pts_target)


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

# %% Compute covariances
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
#V[:, 1] = -V[:, 1]
#U [:, 1] = -U[:, 1]
C_chk = U @ np.diag(S) @ V.T
print('Check decomposition: ', np.linalg.norm(C_chk - C))

# %%
#R = V.T @ U
R = V @ U.T
t = cog_target - R @ cog_model

# %%
T_str = ""
for i in range(3):
    T_str += str(R[i, 0]) + " " + str(R[i, 1]) + " " + str(R[i, 2]) + " " + str(t[i]) + " "
T_str += "0 0 0 1"
print(str(cog_target[0]) + " " + str(cog_target[1]) + " " + str(cog_target[2]))
print(T_str)

# %% Apply transformation
pts_model_hat = (R @ pts_model.T).T + t

# %% Visualization
plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
ax.scatter3D(pts_model[:,0], pts_model[:,1], pts_model[:,2], c='r', s=25)
ax.scatter3D(pts_model_hat[:,0], pts_model_hat[:,1], pts_model_hat[:,2], c='g', s=25)
ax.scatter3D(pts_target[:,0], pts_target[:,1], pts_target[:,2], c='b', s=25)
set_3d_axes_equal(ax)

# %% Residuals
diff = np.linalg.norm(pts_model_hat - pts_target, axis=1)
print('RMSE: ', np.sqrt(np.sum(np.power(diff, 2)) / diff.size))
