import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def set_3d_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_fustrum(ax, X, R, f=1, scale=1, img_limits=[1, 1], w=1, h=1, c='r'):
    cam_dir = -scale * R @ np.array([[0, 0, f]]).T
    cam_dir_line = np.array([[X[0], X[0]+cam_dir[0, 0]], [X[1], X[1]+cam_dir[1, 0]], [X[2], X[2]+cam_dir[2, 0]]]).T

    ax.plot3D(X[0], X[1], X[2], c=c, marker='o')
    ax.plot3D(cam_dir_line[:, 0], cam_dir_line[:, 1], cam_dir_line[:, 2], 'gray')

    w = img_limits[0]
    h = img_limits[1]
    f_pt_0 = np.array([[-w, -h, 0], [-w, h, 0], [w, h, 0], [w, -h, 0], [-w, -h, 0]])
    fustrum = scale * (R @ f_pt_0.T ).T + np.array([X[0] - cam_dir[0, 0], X[1] - cam_dir[1, 0], X[2] - cam_dir[2, 0]])
    ax.plot3D(fustrum[:,0], fustrum[:,1], fustrum[:,2], c=c)
    for pt in fustrum:
        ax.plot3D([X[0], pt[0]], [X[1], pt[1]], [X[2], pt[2]], c=c)

def plot_crs(ax, crs, X=[0, 0, 0]):
    ax.plot3D([X[0], X[0]+crs[0, 0]], [X[1], X[1]+crs[0, 1]], [X[2], X[2]+crs[0, 2]], c='r')
    ax.plot3D([X[0], X[0]+crs[1, 0]], [X[1], X[1]+crs[1, 1]], [X[2], X[2]+crs[1, 2]], c='g')
    ax.plot3D([X[0], X[0]+crs[2, 0]], [X[1], X[1]+crs[2, 1]], [X[2], X[2]+crs[2, 2]], c='b')
