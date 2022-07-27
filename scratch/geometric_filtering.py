# %%
#%matplotlib qt
import os
import sys
from turtle import color
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.spatial.transform import Rotation

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from materials.photo_fundamental_mx import skew, draw_epilines
from utils.viz.viz import plot_fustrum, plot_crs, set_3d_axes_equal
from utils.uni_io.exif import get_exif_data
from utils.nb import isnotebook

# %% Input data
#processing_folder = "/media/ext/test/b9d2eca6-c65d-4d15-9ad0-5e67c3b6c66b"
#processing_folder = "/media/ext/test/3dedb099-594d-4184-867f-a8def2995228"
processing_folder = "/media/ext/test/18280884-b94b-4e13-aec2-aa04343f3940"
sfm_data = processing_folder + "/cluster_0/results/matches/sfm_data.json"
if_scaled = True

original_image_folder = '/cluster_0/input_images/'
if if_scaled:
    image_folder = original_image_folder + '/scaled_images/'
else:
    image_folder = original_image_folder

# %% Read SFM data file
with open(sfm_data, 'r') as json_file:
    data = json.load(json_file)

# %%
n_view = len(data['views'])
# def get_view_data(idx):
#     view = data['views'][idx]['value']['ptr_wrapper']['data']
#     X = np.array(view["center"])
#     R = np.array(view["rotation"])
#     w = view['width']
#     h = view['height']
#     f = h
#     #K = np.array([[f, 0, w/2.0], [0, f, h/2.0], [0, 0, 1]])
#     #K = np.array([[3666.666504, 0, 2432], [0, 3666.666504, 1824], [0, 0, 1]])
#     den = 2.0 if if_scaled else 1.0
#     #K = np.array([[3666.666504/den, 0, 2432/den], [0, 3666.666504/den, 1824/den], [0, 0, 1]])
#     K = np.array([[2134.9, 0, 1368], [0, 2134.9, 912], [0, 0, 1]])
#     return X, R, K, view

# idx1 = 43
# idx2 = 44

idx1 = 43
idx2 = 21

# idx1 = 6
# idx2 = 7

# idx1 = 27
# idx2 = 28

def get_view_data(idx):
    view = data['views'][idx]['value']['ptr_wrapper']['data']
    img_path = processing_folder + original_image_folder + view['filename']
    exif_data = get_exif_data(img_path)
    view['yaw'] = exif_data['Flight Yaw Degree']
    view['pitch'] = exif_data['Flight Pitch Degree']
    view['roll'] = exif_data['Flight Roll Degree']
    yaw_g = exif_data['Gimbal Yaw Degree']
    pitch_g = exif_data['Gimbal Pitch Degree']
    roll_g = exif_data['Gimbal Roll Degree']
    X = np.array(view["center"])

    R_f = Rotation.from_euler('zyx', [view['yaw'], view['pitch'], view['roll']], degrees=True).as_matrix()
    R90_f = Rotation.from_euler('xyz', [-180, 0, 0], degrees=True).as_matrix()
    R_g = Rotation.from_euler('zyx', [yaw_g, pitch_g, roll_g], degrees=True).as_matrix()
    R90_g = Rotation.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix()
    R = R_f @ R90_f
    #R = R_f @ R90_g @ R_g

    #R = Rotation.from_euler('zxz', [view['yaw'], view['pitch'], view['roll']], degrees=False).as_matrix()
    #R = Rotation.from_euler('xyz', [view['pitch'], view['roll'], view['yaw']], degrees=True).as_matrix()
    # R = Rotation.from_euler('zxz', [view['yaw'], view['pitch'], view['roll']], degrees=False).as_matrix()
    R = np.array(view["rotation"])

    w = exif_data['Exif Image Width']
    h = exif_data['Exif Image Height']
    den = 2.0 if if_scaled else 1.0
    f_metric = float(exif_data['Focal Length'].split('mm')[0])
    pix_width = 13.20
    f_pix = w / pix_width * f_metric
    print(view['filename'])
    print(w, pix_width, f_metric, f_pix)
    print(view['yaw'], view['pitch'], view['roll'])
    print([yaw_g, pitch_g, roll_g])
    print(' ')
    K = np.array([[f_pix/den, 0, w/den/2.0], [0, f_pix/den, h/den/2.0], [0, 0, 1]])
    return X, R, K, view

# %%
if isnotebook() and False:
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
# idx1 = 0
# idx2 = 1

# idx1 = 26
# idx2 = 27

# idx1 = 27
# idx2 = 28

# idx1 = 57
# idx2 = 58

X1, R1, K1, view1 = get_view_data(idx1)
t1 = -R1 @ X1
print(view1['yaw'], view1['pitch'], view1['roll'])

X2, R2, K2, view2 = get_view_data(idx2)
t2 = -R2 @ X2
print(view2['yaw'], view2['pitch'], view2['roll'])

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
t = R1 @ np.array(X2 - X1) # = t1 - dR.transpose() @ t2
#t = t2 - dR @ t1 # = R2 @ np.array(X1 - X2)
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
image_path_1 = processing_folder + image_folder + view1['filename']
img_1 = cv2.imread(image_path_1, 0)
image_path_2 = processing_folder + image_folder + view2['filename']
img_2 = cv2.imread(image_path_2, 0)


# %%
def update_view(pts):

    if False:
        epilines = (F @ pts.transpose()).transpose()
        epilines /= np.linalg.norm(epilines[:, :2], axis=1).reshape(-1, 1) # optional normalization of A, B
    else:
        if pts.shape[1] == 3:
            pts = np.array([[pts[0][0], pts[0][1]]])
        epilines = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), 2, F)
        epilines = epilines.reshape(-1, 3)

    img_1_epi, img_2_epi = draw_epilines(img_1, img_2, epilines, pts, color=[255, 0, 0])
    return img_1_epi, img_2_epi

if isnotebook():
    test_pts = np.array([[750, 750, 1]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    img_1_epi, img_2_epi = update_view(test_pts)
    ax1.imshow(img_1_epi)
    ax2.imshow(img_2_epi)

else:

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_1, picker=True)
    ax2.imshow(img_2)
    def onpick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
        pts = np.array([[int(event.xdata), int(event.ydata), 1]])
        img_1_epi, img_2_epi = update_view(pts)
        ax1.imshow(img_1_epi)
        ax2.imshow(img_2_epi)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onpick)
    mngr = plt.get_current_fig_manager()
    mngr.window.showMaximized()
    plt.show()

# %%
# if isnotebook():
#     # 81.0474 3.94483 109.512 798.575 493.582
#     pts1 = np.array([[81, 3, 1]])
#     #pts2 = np.array([[109.512, 798.575, 1]])
#     pts2 = np.array([[1093.58, 796.228, 1]])
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#     img_1_epi, img_2_epi = update_view(pts1)
#     ax1.imshow(img_1_epi)
#     #img1 = cv.circle(img1, (pts2[0], pts2[1]),50,color_fn,50)
#     ax2.imshow(img_2_epi)
#     ax2.scatter(pts2[:, 0], pts2[:, 1], s=50, color='g')

#     # %%
#     F_x = (F @ pts1.transpose()).transpose()
#     res = np.dot(F_x, pts2.transpose())**2 / np.linalg.norm(F_x[:, :2], axis=1)**2
#     print(res)

#     # %%
#     sift = cv2.SIFT_create()

#     # find the keypoints and descriptors with SIFT
#     kps1, des1 = sift.detectAndCompute(img_1, None)
#     kps2, des2 = sift.detectAndCompute(img_2, None)
#     print("Extraction is done.")

#     # %%
#     def extract_points_from_keypoints(kps):
#         kp_pts = np.zeros((len(kps), 2))
#         for idx, kp in enumerate(kps):
#             kp_pts[idx, :] = [kp.pt[0], kp.pt[1]]
#         return kp_pts

#     kp_pts_1 = extract_points_from_keypoints(kps1)
#     kp_pts_2 = extract_points_from_keypoints(kps2)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#     ax1.scatter(kp_pts_1[:, 0], kp_pts_1[:, 1], s=1, color='g')
#     ax2.scatter(kp_pts_2[:, 0], kp_pts_2[:, 1], s=1, color='g')

#     # %%
#     # reporting_idx = int(len(kps1)/100)
#     # if idx_1 % reporting_idx == 0:
#     #     print(idx_1 / len(kps1) * 100)

#     def draw_epilines2(img1, img2, epilines, pts, color=None):
#         """img1 - image on witch we draw the epilines for the points in img2
#         lines - corresponding epilines"""
#         r = img1.shape[0]
#         c = img1.shape[1]
#         # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#         # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
#         for r, pt in zip(epilines,pts):
#             #r = r[0]
#             if color == None:
#                 color_fn = tuple(np.random.randint(0,255,3).tolist())
#             else:
#                 color_fn = color

#             x0,y0 = map(int, [0, -r[2]/r[1] ])
#             x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#             #print(x0, y0, x1, y1)
#             print(pt)
#             img2 = cv2.line(img2, (x0,y0), (x1,y1), color_fn,25)
#             #img1 = cv.circle(img1, (pt[0], pt[1]),50,color_fn,50)
#             img1 = cv2.circle(img1, (pt[0], pt[1]),25,color_fn,25)
#         return img1, img2

#     img_1_epi = cv2.imread(image_path_1)
#     img_2_epi = cv2.imread(image_path_2)

#     #for idx_1 in range(0, kp_pts_1.shape[0]):
#     pts_inside = []
#     for idx1 in [50000]:
#         pt_1 = kp_pts_1[idx1, :]
#         pt_1_h = np.array([pt_1[0], pt_1[1], 1.0])

#         epilines = (F @ np.array([pt_1_h]).transpose()).transpose()
#         epilines /= np.linalg.norm(epilines[:, :2], axis=1).reshape(-1, 1) # optional normalization of A, B
#         img_1_epi, img_2_epi = draw_epilines2(img_1_epi, img_2_epi, epilines, [pt_1.astype('int')], color=[255, 0, 0])

#         F_x = (F @ pt_1_h.transpose()).transpose()

#         for kp2 in kps2:
#             pt_2_h = np.array([kp2.pt[0], kp2.pt[1], 1.0])
#             res = np.dot(F_x, pt_2_h.transpose())**2 / np.linalg.norm(F_x[:2])**2
#             if res < 100**2:
#                 pts_inside.append([kp2.pt[0], kp2.pt[1]])

#     pts_inside = np.array(pts_inside)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#     ax1.imshow(img_1_epi)
#     ax2.scatter(pts_inside[:, 0], pts_inside[:, 1], s=1, color='g')
#     ax2.imshow(img_2_epi)
