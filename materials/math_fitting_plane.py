# %% External imports

%matplotlib qt

import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# %% Internal imports
__file__ = '/home/zoltan/Repo/uni/scratch/wall_densification.py'
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from utils.viz.viz import set_3d_axes_equal

# %%
# point_file_path = '/home/zoltan/Downloads/wall-wesepe/sketch_plane.txt'
# point_file_data = np.loadtxt(point_file_path, delimiter="," , dtype=str)
# points = [[float(point_file_data[k, 1]), float(point_file_data[k, 2]), float(point_file_data[k, 3])] for k in range(len(point_file_data))]
# points = np.array(points)
# point_ids = [point_file_data[k, 0] for k in range(len(point_file_data))]

point_file_path = '/home/zoltan/Repo/pix/.scratch/task0/e5972c03-2719-43bb-92c9-6b453c0166bd/meta_tile_group_0_8fcba91c-ccf1-4c89-bd48-745c50fb57df/wall_points_2.txt'
point_file_data = np.loadtxt(point_file_path)
print(point_file_data)
points = point_file_data
points = np.vstack((points, points[0, :].reshape(1, -1)))
point_ids = list(range(0, len(points)))

idx = list(range(len(point_ids)))
# idx = []
# for k in range(len(point_ids)):
#     if int(point_ids[k]) in [9, 10, 29, 11, 25, 26, 27, 28, 30, 31]:
#           idx.append(k)

plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
ax.scatter3D(points[idx,0], points[idx,1], points[idx,2], c='r', s=50)
ax.plot3D(points[idx,0], points[idx,1], points[idx,2], '-')

for k in range(len(point_ids)):
#    if k in idx:
        ax.text(points[k,0], points[k,1], points[k,2], point_ids[k])

ax.set_title("Points in input coordinate system")
ax.set_xlabel('X'); ax.set_ylabel('Y')
set_3d_axes_equal(ax)
# %%
centroid = np.mean(points, axis=0)
points_c = points - centroid
#points_c = np.array(points)

# %%
import scipy
from scipy.spatial.transform import Rotation

R_local = Rotation.from_euler('xyz', [45.0, 45.0, 45.0], degrees=True).as_matrix()
points_c_r =  points_c @ R_local

if False:
    A = np.c_[points_c_r[:,0], points_c_r[:,1], np.ones(points_c_r.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, points_c_r[:,2])
    plane = lambda X, Y: C[0]*X + C[1]*Y + C[2]
    # c1*x + c2y + c3 = z
    # c3 should be 0 because centroid coordinates, and thus x1*x + c2*y - z = 0
    n = [C[0], C[1], -1]
else:
    # using SVD
    D = points_c_r.T @ points_c_r
    [U, S , V] = np.linalg.svd(D)
    print('S=', S)
    n = U[:, -1]
    print('n=', n)
    # a(x-x0) + b(y-y0) + c(z-z0) = 0, where n = [a, b, c]
    # if (x0, y0, z0) = (0,0,0), then z = (-ax-by)/c
    plane = lambda X, Y: (-n[0]*X -n[1]*Y) / n[2]

minx, maxx = points_c_r[:, 0].min(), points_c_r[:, 0].max()
miny, maxy = points_c_r[:, 1].min(), points_c_r[:, 1].max()
minz, maxz = points_c_r[:, 2].min(), points_c_r[:, 2].max()

# resampling points
X,Y = np.meshgrid(np.arange(minx, maxx, 0.5), np.arange(miny, maxy, 0.5))
Z = plane(X, Y)
z_plane_r = plane(points_c_r[:, 0], points_c_r[:, 1])

check_normal = 0
for k in range(len(points_c_r)) :
    p_plane = np.array([points_c_r[0, 0] - points_c_r[1, 0],
                        points_c_r[0, 1] - points_c_r[1, 1],
                        z_plane_r[0] - z_plane_r[1]])

    check_normal = np.abs(np.dot(p_plane, n) / (np.linalg.norm(p_plane) * np.linalg.norm(n)))

print('check_normal', check_normal)

# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter3D(points_c_r[:,0], points_c_r[:,1], points_c_r[:,2], c='r', s=50)
#ax.scatter3D(points_c_r[:,0], points_c_r[:,1], z_plane_r, c='b', s=50)
ax.plot3D([0, n[0]*5],
          [0, n[1]*5],
          [0, n[2]*5], 'b-')

ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.set_zlim([minz, maxz])

ax.set_title("Points in input coordinate system")
ax.set_xlabel('X'); ax.set_ylabel('Y')
set_3d_axes_equal(ax)

# %%
#xx = np.sqrt(points_c_r[:,0]**2 + points_c_r[:,1]**2)
points_c_r_p = points_c_r.copy()
#points_c_r_p[:, 2] = z_plane_r
points_c_r_p = points_c_r_p @ U

print(np.linalg.norm(points_c_r_p - points_c_r_p @ U @ np.linalg.inv(U)))

print('chk z (~=0): ', np.sum(np.abs(points_c_r_p[:, 2])))

plt.plot(points_c_r_p[:, 0], points_c_r_p[:, 1])

# %%
from shapely import Polygon, Point
poly = Polygon(points_c_r_p)
#poly.exterior.coords

wall_points_c_r_p = []
DISTANCE_BETWEEN_GENERATE_POINTS = 0.1
for wall_pt_x in np.arange(poly.bounds[0], poly.bounds[2], DISTANCE_BETWEEN_GENERATE_POINTS):
    for wall_pt_y in np.arange(poly.bounds[1], poly.bounds[3], DISTANCE_BETWEEN_GENERATE_POINTS):
        pt = Point((wall_pt_x, wall_pt_y))
        if poly.contains(pt):
            wall_points_c_r_p.append([wall_pt_x, wall_pt_y])

wall_points_c_r_p = np.array(wall_points_c_r_p)
# %%
plt.plot(points_c_r_p[:, 0], points_c_r_p[:, 1])
plt.scatter(wall_points_c_r_p[:, 0], wall_points_c_r_p[:, 1], s=5)
plt.axis('equal')

# %%
nzv = np.zeros((wall_points_c_r_p.shape[0], 1))
wall_points_c_r = np.hstack((wall_points_c_r_p, nzv)) @ np.linalg.inv(U)
wall_points_c = wall_points_c_r @ R_local.T

points_c_r_chk = points_c_r_p @ np.linalg.inv(U)
points_c_chk = points_c_r_chk @ R_local.T

print('chk (~=0): ', np.linalg.norm(points_c_r_chk - points_c_r))
print('chk (~=0): ', np.linalg.norm(points_c_chk - points_c))
# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter3D(points_c_r[:,0], points_c_r[:,1], points_c_r[:,2], c='r', s=50)
ax.scatter3D(points_c_r_chk[:,0], points_c_r_chk[:,1], points_c_r_chk[:,2], c='g', s=50)
ax.plot3D(points_c_r[:,0], points_c_r[:,1], points_c_r[:,2], 'r-')
ax.plot3D(points_c_r_chk[:,0], points_c_r_chk[:,1], points_c_r_chk[:,2], 'g--')
ax.scatter3D(wall_points_c_r[:,0], wall_points_c_r[:,1], wall_points_c_r[:, 2], c='b', s=50)
ax.set_xlim([minx, maxx])
ax.set_ylim([miny, maxy])
ax.set_zlim([minz, maxz])

# %%
plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
ax.scatter3D(points_c[idx,0], points_c[idx,1], points_c[idx,2], c='r', s=50)
ax.plot3D(points_c[idx,0], points_c[idx,1], points_c[idx,2], '-')
ax.scatter3D(wall_points_c[:,0], wall_points_c[:,1], wall_points_c[:,2], c='b', s=10)
set_3d_axes_equal(ax)

# %%
wall_points_c += centroid

