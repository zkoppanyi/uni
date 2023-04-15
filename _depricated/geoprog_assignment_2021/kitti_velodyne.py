# %% Imports
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# %% External imports

# Library from Lee Clement [lee.clement@robotics.utias.utoronto.ca]
# link: https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_odometry.py
import pykitti

# %% Internal imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from utils.nb import isnotebook

# %% Loading
basedir = '/media/ext/data-uni/kitti-odo'
sequence = '08'

# %% Load Velodyne
dir_velo = basedir + '/data_odometry_velodyne/dataset'
dataset_velo = pykitti.odometry(dir_velo, sequence)

# %% Load camera
dir_img = basedir + '/data_odometry_gray/dataset'
dataset_img = pykitti.odometry(dir_img, sequence)

# %%

def plot_frame(ax1, ax2, velo_data, img_data):
    velo_range = range(0, velo_data.shape[0], 5)
    #ax1.scatter(velo_data[velo_range, 0], velo_data[velo_range, 1], s=2, c=velo_data[velo_range, 3])
    ax1.scatter(velo_data[velo_range, 0], velo_data[velo_range, 1],
                    s=1, c=velo_data[velo_range, 3])
    ax1.set_xlim([-50, 50])
    ax1.set_ylim([-20, 20])
    ax1.set_aspect(1)
    ax2.imshow(img_data, cmap='gray')

if isnotebook():
    velo_data = dataset_velo.get_velo(1)
    img_data = dataset_img.get_cam1(1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))
    plot_frame(ax1, ax2, velo_data, img_data)

else:
    # from matplotlib.animation import FuncAnimation

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))
    # def animate(frame_num):
    #     velo_data = dataset_velo.get_velo(frame_num)
    #     img_data = dataset_img.get_cam1(frame_num)
    #     ax1.clear()
    #     ax2.clear()
    #     plot_frame(ax1, ax2, velo_data, img_data)
    #     return ax1, ax2
    # anim = FuncAnimation(fig, animate, frames=len(dataset_velo), interval=1)
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))
    for i in range(len(dataset_velo)):
        velo_data = dataset_velo.get_velo(i)
        img_data = dataset_img.get_cam1(i)

        ax1.clear()
        ax2.clear()
        plot_frame(ax1, ax2, velo_data, img_data)
        ax1.set_title('Frame #' + str(i))
        #plt.savefig('/home/zoltan/Downloads/video/img_' + str(i) + '.png')
        plt.pause(0.01)

# %%
