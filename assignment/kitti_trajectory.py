# %% Imports
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %% External imports

# Library from Lee Clement [lee.clement@robotics.utias.utoronto.ca]
# link: https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_odometry.py
import pykitti

# %% Loading
basedir = '/media/ext/data-uni/kitti-odo/data_odometry_calib/dataset'
sequence = '08'
#dataset = pykitti.odometry(basedir, sequence, frames=range(0, 20, 1))
dataset = pykitti.odometry(basedir, sequence)

# %%
xyz = []
for pose in dataset.poses:
    R = np.array(pose[:3, :3])
    t = np.array(pose[:3, 3])
    #xyz.append(-np.linalg.inv(R) @ t)
    xyz.append(pose[:3, 3])
xyz = np.array(xyz)

plt.scatter(xyz[:, 0], xyz[:, 2], c='r', s=0.1)
plt.axis('equal')

# %% Height profile
#xy = np.sqrt(np.power(xyz[:, 0], 2) + np.power(xyz[:, 2], 2))
plt.scatter(xyz[:, 0], xyz[:, 1], c='r', s=0.1)
plt.axis('equal')

