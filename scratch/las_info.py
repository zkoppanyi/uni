# %%
import laspy

# %%
las_file_path = '/home/zoltan/Repo/pix/sandbox/input/mcc/lidar.las'
pts = []
with laspy.open(las_file_path) as las:
    print(f"Point format:       {las.header.point_format}")
    print(f"Number of points:   {las.header.point_count}")
    print(f"Number of vlrs:     {len(las.header.vlrs)}")
    print(f"Scales:             {las.header.scales}")

# %%
