# %% Imports
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from utils.viz.viz import plot_fustrum, plot_crs, set_3d_axes_equal

# %% Test angles
euler_deg = [45, 15, -25]

# %% Conversion: Euler angles to rotation matrix
R = Rotation.from_euler('xyz', euler_deg, degrees=True)
print(R.as_matrix())

# %%
##  Intrinsic & extrinsic rotations

# %% Vizualization
def plot_rotation(x, y, z, intrinsic=True):
    plt.figure(figsize=(12,10))
    ax = plt.axes(projection='3d')
    if intrinsic:
        R = Rotation.from_euler('XYZ', [x, y, z], degrees=True)
    else:
        R = Rotation.from_euler('xyz', [x, y, z], degrees=True)
    plot_fustrum(ax, [0, 0, 0], R.as_matrix(), img_limits=[1, 0.5], f=2.0, scale=1.0, c='k')
    plot_crs(ax, np.eye(3), X=[-2, -2, 0])
    crs = R.as_matrix()
    plot_crs(ax, crs)
    set_3d_axes_equal(ax)

plot_rotation(0, 0, 0)

# %% Rotation around X axis (red)
plot_rotation(90, 0, 0)

# %% Rotation around Y axis in the rotated system
# This is the rotation around a downward looking Z axis in the world
plot_rotation(90, 90, 0)

# %% Rotation around Z axis in the double rotated system
# This is the rotation around the X axis in the world
plot_rotation(90, 90, 45)

# %% Explanation
# link: https://en.wikipedia.org/wiki/Euler_angles#Definition_by_intrinsic_rotations
# Twelve combinations of x, y, z:
# Used in mechatrocins: z-x-z, x-y-x, y-z-y, z-y-z, x-z-x, y-x-y
# Navigation: x-y-z, y-z-x, z-x-y, x-z-y, z-y-x, y-x-z

# %% Looking at the specs of scipy:
# link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
# {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or
# {‘x’, ‘y’, ‘z’} for extrinsic rotations.

# %% Ok let's play the same just with extrinsic rotation
plot_rotation(90, 0, 0, intrinsic=False)
# this is the same as before

# %% Rotation around Y axis in the (!!!) world system
plot_rotation(90, 90, 0, intrinsic=False)
# this is different

# %% Rotation around Z axis in the (!!!) world system
plot_rotation(90, 90, 45, intrinsic=False)

# %%
# So we have 12 combinations for intrinsic and 12 combinations for extrinics
# This is all together 24 combinations to describe rotations with Euler angles.

# %% Statement 1
# A sequence of intrinsic rotations produces the same total rotations
# as the sequence of extrinsic rotations with the same angles but in reverse order.
# Proof: https://math.stackexchange.com/questions/1137745/proof-of-the-extrinsic-to-intrinsic-rotation-transform/3314025
rpy = euler_deg
R_int = Rotation.from_euler('xyz', rpy, degrees=True) # extrinsic
ypr = [euler_deg[2], euler_deg[1], euler_deg[0]]
R_ext = Rotation.from_euler('ZYX', ypr, degrees=True) # intrinsic
chk = np.linalg.norm(R_int.as_matrix() - R_ext.as_matrix())
print(f"Difference of the two matrices: {chk}")

# %% Composing rotation matrix from euler angles
# 1. Order of angles
# 2. Order of multiplying matrices

R_x = Rotation.from_euler('x', euler_deg[0], degrees=True).as_matrix()
R_y = Rotation.from_euler('y', euler_deg[1], degrees=True).as_matrix()
R_z = Rotation.from_euler('z', euler_deg[2], degrees=True).as_matrix()

R_int = Rotation.from_euler('XYZ', euler_deg, degrees=True).as_matrix()
chk = np.linalg.norm(R_x @ R_y @ R_z - R_int)
print(f'Intrinsics check: {chk}')

R_ext = Rotation.from_euler('xyz', euler_deg, degrees=True).as_matrix()
chk = np.linalg.norm(R_z @ R_y @ R_x - R_ext)
print(f'Extrinsics check: {chk}')

# %%
## Conversion: rotiation matrix to Euler angles

# %%
R = Rotation.from_euler('xyz', euler_deg, degrees=True)
print(R.as_matrix())
print(R.as_euler('xyz', degrees=True))

# %% Gimbal lock
# link: https://matthew-brett.github.io/transforms3d/gimbal_lock.html
R = Rotation.from_euler('xyz', [10, -90, 10], degrees=True)
print(R.as_matrix())
print(R.as_euler('xyz', degrees=True))

# %%
import math
from math import sin, cos

def getR(pitch, roll, yaw):
    R1 = [cos(yaw)*cos(roll)+sin(yaw)*sin(pitch)*sin(roll), -sin(yaw)*cos(roll)+cos(yaw)*sin(pitch)*sin(roll), -cos(pitch)*sin(roll)]
    R2 = [cos(yaw)*sin(roll)-sin(yaw)*sin(pitch)*cos(roll), -sin(yaw)*sin(roll)-cos(yaw)*sin(pitch)*cos(roll), cos(pitch)*cos(roll)]
    R3 = [-sin(yaw)*cos(pitch), -cos(yaw)*cos(pitch), -sin(pitch)]
    return np.vstack([R1, R2, R3])


yaw = 35/180*math.pi
pitch = -45/180*math.pi
roll = -185/180*math.pi

R_fn = getR(pitch, roll, yaw)
R = Rotation.from_euler('zxz', [yaw, pitch, roll], degrees=False).as_matrix()

print(np.linalg.norm(R_fn-R))
np.set_printoptions(suppress=True)
print(R_fn)
print(' ')
print(R)

# %%


# %%
print(Rotation.from_euler('XYZ', [90, 45, 0], degrees=True).as_matrix())
# %%
print(np.linalg.det(R1))
print(R1.T - np.linalg.inv(R1))

# %%
R1.T - np.linalg.inv(R1)

# %%
