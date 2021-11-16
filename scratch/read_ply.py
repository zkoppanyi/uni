# %% External imports
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d

# %%
ply_file_path = '/home/zoltan/Repo/pix/sandbox/output/normals.ply'
with open(ply_file_path, 'rb') as f:
    plydata = PlyData.read(f)

# %%
pcd = o3d.io.read_point_cloud(ply_file_path)
#o3d.visualization.draw_geometries([pcd])

# %%
n_pts = len(pcd.normals)
#n_pts = 100

raw_normals = list(pcd.normals)[:n_pts]
raw_normals = np.array(raw_normals)
idx = ~np.isnan(raw_normals[:, 0])
raw_normals = raw_normals[idx, :]

raw_points = list(pcd.points)[:n_pts]
raw_points = np.array(raw_points)
#raw_points = raw_points - np.mean(raw_points, axis=0)
raw_points = raw_points[idx, :]

cog = np.mean(raw_points, axis=0)

# %%
n_pts = raw_points.shape[0]
p1 = raw_points
p2 = raw_points + raw_normals * 0.1
points = np.concatenate((p1, p2), axis=0)
points = o3d.utility.Vector3dVector(points)

lines = np.concatenate((
            (np.array(range(0, n_pts-1)).reshape(-1,1).astype(int),
            (np.array(range(0, n_pts-1)) + n_pts).reshape(-1,1).astype(int))),
            axis=1)

line_set = o3d.geometry.LineSet(
    points=points,
    lines=o3d.utility.Vector2iVector(lines),
)

colors = [[1, 0, 0] for i in range(len(lines))]
line_set.colors = o3d.utility.Vector3dVector(colors)

# %%
lidar_station = np.array([542458.625, 127504.828125, 156.346984863])
vec = np.array([[0, 0, 0], [0, 0, 100]])
vec_line = [[0, 1]]
vec_line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(vec + lidar_station),
    lines=o3d.utility.Vector2iVector(vec_line),
)
colors = [[0, 1, 0] for i in range(len(vec_line))]
vec_line_set.colors = o3d.utility.Vector3dVector(colors)

# %%
o3d.visualization.draw_geometries([pcd, line_set, vec_line_set])
#vis = o3d.visualization.Visualizer()
#vis.create_window()
#vis.run()
#vis.destroy_window()



# %%
