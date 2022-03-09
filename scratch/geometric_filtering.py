# %%
#%matplotlib qt
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from materials.photo_fundamental_mx import skew, draw_epilines
from utils.viz.viz import plot_fustrum, plot_crs, set_3d_axes_equal
from utils.nb import isnotebook

# %% Input data
processing_folder = "/media/ext/test/b9d2eca6-c65d-4d15-9ad0-5e67c3b6c66b"
#processing_folder = "/media/ext/test/3dedb099-594d-4184-867f-a8def2995228"
sfm_data = processing_folder + "/cluster_0/results/matches/sfm_data.json"

# %% Read SFM data file
with open(sfm_data, 'r') as json_file:
    data = json.load(json_file)

# %%
n_view = len(data['views'])
def get_view_data(idx):
    view = data['views'][idx]['value']['ptr_wrapper']['data']
    X = np.array(view["center"])
    R = np.array(view["rotation"])
    w = view['width']
    h = view['height']
    f = h
    #K = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1]])
    K = np.array([[3666.666504/2.0, 0, 2432/2.0], [0, 3666.666504/2.0, 1824/2.0], [0, 0, 1]])
    #K = np.array([[3666.666504, 0, 2432], [0, 3666.666504, 1824], [0, 0, 1]])
    return X, R, K, view

# %%
if isnotebook():
    traj = []
    R_cams = []
    t_cams = []
    for k in range(0, n_view):
        X, R, _, _ = get_view_data(k)
        R_cams.append(R)
        t_cams.append(-R @ X)
        traj.append(X)
    traj = np.array(traj)
    # plt.scatter(traj[:, 0], traj[:, 1])
    # plt.axis('equal')

    fig = plt.figure(figsize=(12,10))
    ax = plt.axes(projection='3d')
    for k in range(len(R_cams)):
        ax.plot3D(traj[k, 0], traj[k, 1], traj[k, 2], c='g', marker='o')
        X = -R_cams[k].T @ t_cams[k]
        plot_fustrum(ax, X, R_cams[k], f=1.0, scale=10)
    set_3d_axes_equal(ax)
    plt.show()

# %% Relative camera motion
idx1 = 1
idx2 = 2

# idx1 = 26
# idx2 = 27

# idx1 = 27
# idx2 = 28

X1, R1, K1, view1 = get_view_data(idx1)
t1 = -R1 @ X1
X2, R2, K2, view2 = get_view_data(idx2)
t2 = -R2 @ X2

if True:
    ax = plt.axes(projection='3d')
    plot_fustrum(ax, X1, R1, f=1.0, scale=10)
    plot_fustrum(ax, X2, R2, f=1.0, scale=10)
    set_3d_axes_equal(ax)
    plt.show()

# %%
dR = R2 @ R1.transpose()

P = [40, 40, 100]
p1 = R1 @ (P - X1)
p2 = R2 @ (P - X2)
t = R1 @ np.array(X2 - X1)
#t = t2 - dR @ t1
coplan_const = (p1-t).transpose() @ np.cross(t, p1)
print("Point is coplanar: ", coplan_const)

dR = R2 @ R1.transpose()
p2_chk = dR @ (p1 - t)
print(p2_chk - p2)
print("Should be 0: ", (p2.transpose() @ dR) @ np.cross(t, p1))
print("Should be 0: ", (p2.transpose() @ dR) @ skew(t) @ p1)
E = (dR @ skew(t))
print("Should be 0: ", p2.transpose() @ E @ p1)
print("E: ", E)

print("rank(E) == 2: ", np.linalg.matrix_rank(E))
print("det(E) == 0: ", np.linalg.det(E))
d = 2 * E @ E.transpose() @ E - np.trace(E @ E.transpose()) * E
print("def.: == 0: ", np.linalg.norm(d))

print("E", E)

F = np.linalg.inv(K2).transpose() @ E @ np.linalg.inv(K1)
print("F", F)

# %%
#image_folder = '/cluster_0/input_images/scaled_images/'
image_folder = '/cluster_0/input_images/'
image_path_1 = processing_folder + image_folder + view1['filename']
img_1 = cv2.imread(image_path_1, 0)
image_path_2 = processing_folder + image_folder + view2['filename']
img_2 = cv2.imread(image_path_2, 0)

# %%
def update_view(pts):
    epilines = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), 2, F)
    epilines = epilines.reshape(-1, 3)
    img_1_epi, img_2_epi = draw_epilines(img_1, img_2, epilines, pts, color=[255, 0, 0])
    return img_1_epi, img_2_epi

if isnotebook():
    test_pts = np.array([[750, 750]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    img_1_epi, img_2_epi = update_view(test_pts)
    ax1.imshow(img_1_epi)
    ax2.imshow(img_2_epi)

# %%
if not isnotebook():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_1, picker=True)
    ax2.imshow(img_2)
    def onpick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
        # row = event.ydata
        # col = event.xdata
        #ax1.plot(event.xdata, event.ydata, 'r*')
        #ax2.plot(event.xdata, event.ydata, 'b*')
        pts = np.array([[int(event.xdata), int(event.ydata)]])
        img_1_epi, img_2_epi = update_view(pts)
        ax1.imshow(img_1_epi)
        ax2.imshow(img_2_epi)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onpick)
    mngr = plt.get_current_fig_manager()
    mngr.window.showMaximized()
    plt.show()

