# %% Imports
import itertools
from sqlite3 import Timestamp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pygga import gen_gga
import scipy.interpolate
from scipy.spatial.transform import Rotation

# %% External imports

# Library from Lee Clement [lee.clement@robotics.utias.utoronto.ca]
# link: https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_odometry.py
import pykitti

# %% Loading
basedir = '/media/ext/data-uni/kitti-odo/data_odometry_calib/dataset'
basedir_lidar = '/media/ext/data-uni/kitti-odo'
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

# %% Load Velodyne
dir_velo = basedir_lidar + '/data_odometry_velodyne/dataset'
dataset_velo = pykitti.odometry(dir_velo, sequence)

# %%
data_range = list(range(0, 1500, 25))
t_frame = []
R_frame = []
velo_points = []

n = len(dataset.poses)
plt.figure(figsize=(12,10))
for idx in data_range:
    pose = dataset.poses[idx]
    T_cam0_velo = dataset.calib.T_cam0_velo
    velo_data = dataset_velo.get_velo(idx)

    R = np.array(pose[:3, :3])
    t = np.array(pose[:3, 3])

    velo_range = range(0, velo_data.shape[0], 5)
    velo_frame = velo_data[velo_range, :]
    velo_xyz = velo_frame[:, 0:3]
    velo_points.append(velo_xyz)

    velo_xyz_h = np.hstack((velo_xyz, np.ones((velo_xyz.shape[0], 1))))
    #velo_xyz_t = (R @ velo_xyz.T).T
    point_cam0 = (T_cam0_velo @ velo_xyz_h.T).T
    point_cam0 = point_cam0[:, 0:3]
    velo_xyz_t = (R @ point_cam0.T).T + t

    plt.scatter(velo_xyz_t[:, 0], velo_xyz_t[:, 2],
                    s=1, c=velo_frame[:, 2], cmap='Blues')

    t_frame.append(t)
    R_frame.append(R)

for idx in data_range:
    pose = dataset.poses[idx]
    t = np.array(pose[:3, 3])
    plt.scatter(t[0], t[2], c='r')

# %% Make up timestamps
t_frame = np.array(t_frame)
s = np.cumsum(np.linalg.norm(np.diff(t_frame, axis=0), axis=1))
speed = 60
ts = s / speed / 1000 * 3600
ts = np.append(0, ts)

idx2ts = scipy.interpolate.interp1d(data_range, ts, kind="cubic")
ts_frame = idx2ts(data_range)

# %%
ts_high_res = []
t_high_res = []
R_high_res = []
rpy_high_res = []
R_w_j = dataset.poses[0][:3, :3]

for idx in range(min(data_range), max(data_range)):
    pose = dataset.poses[idx]

    t = np.array(pose[:3, 3])
    t_high_res.append(t)


    R = np.array(pose[:3, :3])
    # R_w_i = np.array(pose[:3, :3])
    # R_i_j = R_w_j.T @ R_w_i @ R_w_j
    # R_w_j = R_w_i
    # R = R_i_j

    R_high_res.append(R)

    rpy = Rotation.from_matrix(R).as_euler('xyz', degrees=True) # extrinsic
    rpy_high_res.append(rpy)

    ts_high_res.append(idx2ts(idx))

t_high_res = np.array(t_high_res)
rpy_high_res = np.array(rpy_high_res)
ts_high_res = np.array(ts_high_res)

int_x = scipy.interpolate.interp1d(ts_high_res, t_high_res[:, 0], kind="cubic", fill_value="extrapolate")
int_y = scipy.interpolate.interp1d(ts_high_res, t_high_res[:, 1], kind="cubic", fill_value="extrapolate")
int_z = scipy.interpolate.interp1d(ts_high_res, t_high_res[:, 2], kind="cubic", fill_value="extrapolate")

int_roll = scipy.interpolate.interp1d(ts_high_res, rpy_high_res[:, 0], kind="cubic", fill_value="extrapolate")
int_pitch = scipy.interpolate.interp1d(ts_high_res, rpy_high_res[:, 1], kind="cubic", fill_value="extrapolate")
int_yaw = scipy.interpolate.interp1d(ts_high_res, rpy_high_res[:, 2], kind="cubic", fill_value="extrapolate")

ts_nav = np.arange(ts_high_res.min(), ts_high_res.max(), step=0.2)
t_nav = np.array([int_x(ts_nav), int_y(ts_nav), int_z(ts_nav)]).T
rpy_nav = np.array([int_roll(ts_nav), int_pitch(ts_nav), int_yaw(ts_nav)]).T

# %%
plt.plot(ts_nav, rpy_nav[:, 0], c='r')
plt.plot(ts_nav, rpy_nav[:, 1], c='g')
plt.plot(ts_nav, rpy_nav[:, 2], c='b')

# %% Show unsync data
plt.scatter(ts_nav, np.ones((len(ts_nav), 1)), c='b')
plt.scatter(ts_frame, np.ones((len(ts_frame), 1)), c='r')
plt.xlim([0, 6])
plt.ylim([0, 2])

# %% Write out the stuff...
output_folder = "./output"
nav_data = np.hstack((ts_nav.reshape(-1, 1), t_nav, rpy_nav))
np.savetxt(output_folder+"/nav.txt", nav_data, header="Timestamp [s], X [m], Y [m], Z [m], Roll [deg], Pitch [deg], Heading [deg]", delimiter=",", fmt="%.3f")

np.savetxt(output_folder+"/T_nav_velo.txt", T_cam0_velo)

# np.savetxt(output_folder+"/velo.txt", ts_frame, header="Timestamp [s]", delimiter=",", fmt="%.3f")
# for idx, velo_frame in enumerate(velo_points):
#     np.savetxt(output_folder+"/frame_" + str(idx) + ".txt", velo_frame, header="X [m], Y [m], Z [m]", delimiter=",", fmt="%.3f")

velo_out_data = []
for idx, velo_frame in enumerate(velo_points):
    velo_out_data.append({
        "timestamp": ts_frame[idx],
        "points": velo_frame
    })

with open(output_folder+"/velo.npy", 'wb') as f:
    np.save(f, velo_out_data)

# %%
