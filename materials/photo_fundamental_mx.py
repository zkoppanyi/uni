# %% Global imports
#%matplotlib qt
import os
import sys
from turtle import color
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# %% Local imports
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from utils.nb import isnotebook
from utils.viz.viz import plot_fustrum, plot_crs, set_3d_axes_equal
from utils.uni_io.bundle_out import read_bundle_out

# %% Helper functions
def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def draw_epilines(img1, img2, epilines, pts, color=None):
    """img1 - image on witch we draw the epilines for the points in img2
       lines - corresponding epilines"""
    r = img1.shape[0]
    c = img1.shape[1]
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt in zip(epilines,pts):
        #r = r[0]
        if color == None:
            color_fn = tuple(np.random.randint(0,255,3).tolist())
        else:
            color_fn = color

        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        #print(x0, y0, x1, y1)
        img2 = cv.line(img2, (x0,y0), (x1,y1), color_fn,50)
        #img1 = cv.circle(img1, (pt[0], pt[1]),50,color_fn,50)
        img1 = cv.circle(img1, (pt[0], pt[1]),50,color_fn,50)
    return img1, img2

# %%
if __name__ == "__main__":

    base_folder = "/media/ext/test/61deef7e-dc28-4f60-a061-172870ba040c"
    img1_path = base_folder + "/cluster_0/input_images/DJI_0065.JPG"
    img2_path = base_folder + "/cluster_0/input_images/DJI_0069.JPG"

    img1_path = base_folder + "/cluster_0/input_images/DJI_0058.JPG"
    img2_path = base_folder + "/cluster_0/input_images/DJI_0063.JPG"

    img1_name = img1_path.split('/')[-1]
    img2_name = img2_path.split('/')[-1]

# %% Load images
    img1 = cv.imread(img1_path, 0)
    img2 = cv.imread(img2_path, 0)

# %% Estimate F
    if False:
        sift = cv.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        print("Extraction is done.")

        # %% FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        print("Matching is done.")

        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i, (m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)

        print("Ratio test is done.")

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F_est, mask = cv.findFundamentalMat(pts1, pts2, cv.RANSAC)
        print("F matrix is done.")

        # We select only inlier points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
    else:
        F_est = np.array([[ 1.06070836e-08,  2.15106458e-07,  1.41253122e-03],
        [-2.06201901e-07,  5.05771082e-09,  1.72317600e-03],
        [-1.63734635e-03, -1.77939716e-03,  1.00000000e+00]])

# %% Compute F from known poses
    if True:
        bundle_out_file = base_folder + "/bundle_8b16700f-c433-4172-8665-fbce4c56d099.out"
        cam, R, tran, [x,y,z], [x_mean, y_mean, z_mean], cam_view_pt, n_view_pt = read_bundle_out(bundle_out_file)

        img1_idx = None
        img2_idx = None
        for i in cam.name:
            cam_name = cam.name[i][0]
            if cam_name == img1_name:
                img1_idx = i
            if cam_name == img2_name:
                img2_idx = i

        R1 = R[:, :, img1_idx]
        t1 = tran[img1_idx] # + [x_mean, y_mean, z_mean]
        X1 = cam.pos[img1_idx]
        R2 = R[:, :, img2_idx]
        t2 = tran[img2_idx] # + [x_mean, y_mean, z_mean]
        X2 = cam.pos[img2_idx]
        dR = R2 @ R1.transpose()
        dt = t2 - dR @ t1
        #dt = dt / np.linalg.norm(dt)

        # It seems that in bundle file the up direction is different
        from scipy.spatial.transform import Rotation
        R90 = Rotation.from_euler('zyx', [0, 0, 180], degrees=True).as_matrix()
        R1 = R90 @ R1
        R2 = R90 @ R2

# %% Visualize stereo pair in 3d space
    if False:
        fig = plt.figure(figsize=(12,10))
        ax = plt.axes(projection='3d')
        #ax.plot3D(x, y, z, 'g.')
        plot_fustrum(ax, X1, R1, f=1.0, scale=10)
        plot_fustrum(ax, X2, R2, f=1.0, scale=10)

        # plot_fustrum(ax, [0, 0, 0], np.eye(3), f=1.0, scale=10)
        # plot_fustrum(ax, dt, R90@dR, f=1.0, scale=10)
        set_3d_axes_equal(ax)
        plt.show()

# %% Derivation
    # Following this: https://www.cs.cmu.edu/~16385/s17/Slides/12.2_Essential_Matrix.pdf

    dR = R2 @ R1.transpose()
    #dt = t2 - dR @ t1
    dt = R1 @ np.array(X2 - X1) # = t1 - dR.transpose() @ t2

    P = [40, -30, 20]
    p1 = R1 @ (P - X1)
    p2 = R2 @ (P - X2)
    coplan_const = (p1-dt).transpose() @ np.cross(dt, p1)
    print("Point is coplanar: ", coplan_const)

    #dR = R2 @ R1.transpose()
    p2_chk = dR @ (p1 - dt)
    print(p2_chk - p2)
    print("Should be 0: ", (p2.transpose() @ dR) @ np.cross(dt, p1))
    print("Should be 0: ", (p2.transpose() @ dR) @ skew(dt) @ p1)
    E = (dR @ skew(dt))
    print("Should be 0: ", p2.transpose() @ E @ p1)
    print("E: ", E)

    print("rank(E) == 2: ", np.linalg.matrix_rank(E))
    print("det(E) == 0: ", np.linalg.det(E))
    d = 2 * E @ E.transpose() @ E - np.trace(E @ E.transpose()) * E
    print("def.: == 0: ", np.linalg.norm(d))

# %% Compose F matrix from E
    def getK(idx):
        f = cam.f[idx]
        xo = float(cam.size_x[idx]) / 2.0 + cam.xo[idx]
        yo = float(cam.size_y[idx]) / 2.0 + cam.yo[idx]
        return np.array([[f, 0, xo], [0, f, yo], [0, 0, 1]])
    K1 = getK(img1_idx)
    K2 = getK(img2_idx)

    E_est = K2.transpose() * F_est * K1

    F = np.linalg.inv(K2).transpose() @ E @ np.linalg.inv(K1)

    E_chk = K2.transpose() @ F @ K1
    print("E-E_chk:", np.linalg.norm(E-E_chk))

# %% Visualization

    def update_view(pts):
        epilines = cv.computeCorrespondEpilines(pts.reshape(-1, 1, 2), 1, F)
        epilines = epilines.reshape(-1, 3)
        # epilines = (F @ pts.transpose()).transpose()
        # epilines /= np.linalg.norm(epilines[:, :2], axis=1).reshape(-1, 1) # optional normalization of A, B
        # # line: A*x + B*Y + C = 0, where lines = [A, B, C]
        img_1_epi, img_2_epi = draw_epilines(img1, img2, epilines, pts, color=[255, 0, 0])
        return img_1_epi, img_2_epi

    if isnotebook():
        # drawing epilines on left image
        test_pts = np.array([[1000, 1000], [2000, 1000]])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        img_1_epi, img_2_epi = update_view(test_pts)
        ax1.imshow(img_1_epi)
        ax2.imshow(img_2_epi)

    else:
        # Interactive point picker
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img1, picker=True)
        ax1.title.set_text("Pick a point on this image!")
        ax2.imshow(img2)
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

