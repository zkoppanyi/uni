# %%
import numpy as np
from pyproj import CRS, Transformer, Proj
from geopy.distance import geodesic

# %%
crs_eov = CRS.from_epsg(23700) # HD72 / EOV
crs_etrs89_llh = CRS.from_epsg(4258) # ETRS89/GRS80, lat/lon
crs_etrs89_xyz = CRS.from_epsg(4936) # ETRS89, ECEF-XYZ

print('EOV/HD72: ')
print(crs_eov.to_wkt(pretty=True))

# %%
eov2etrs89_xyz = Transformer.from_crs(crs_eov, crs_etrs89_xyz)
eov2etrs89_llh = Transformer.from_crs(crs_eov, crs_etrs89_llh)

# %%
# BUTE GPS Point: https://www.gnssnet.hu/pdf/BUTE.pdf
y_eov = 650684.479 # easting
x_eov = 237444.175 # northing
z_eov = 137.177 # EOMA height

p_etrs89_xyz = eov2etrs89_xyz.transform(y_eov, x_eov, z_eov)
p_etrs89_llh = eov2etrs89_llh.transform(y_eov, x_eov, z_eov)

# %% Comparison to eht2: http://eht.gnssnet.hu/index.php
p_eht_etrs89_xyz = np.array([4081882.378, 1410011.142, 4678199.391])
p_eht_etrs89_llh = np.array([47.4809437284, 19.0565297521, 180.811])

print(p_etrs89_xyz - p_eht_etrs89_xyz)

# %% Geodetic distance between two points defined with lat/lon
coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)
sol_1 = geodesic(coords_1, coords_2, ellipsoid='GRS-80').m * 100
sol_2 = geodesic(coords_1, coords_2, ellipsoid='WGS-84').m * 100
print('Difference GRS-80 vs. WGS-84 [cm]: ', sol_1 - sol_2)

# %%
print('Difference XY [cm]: ', geodesic(p_eht_etrs89_llh[0:2], p_etrs89_llh[0:2]).m * 100)
# Note: the Z of p_etrs89_llh[2] is the same as input, no Z coordinate transformation
# print('Difference Z  [m]: ', (p_eht_etrs89_llh[2] - p_etrs89_llh[2]))

# %% Original definiation of HD72/EOV in proj4 format
print('Original definiation:')
print(crs_eov.to_proj4())

# %%
# < transzformáció 1 cm-es középhibán belül azonos az EHT-vel
# az EHT transzformációja 10 cm-es középhibájú
nadgrids = '/home/zoltan/Repo/eov2etrs/etrs2eov_notowgs.gsb'# XY grid
geoidgrids = '/home/zoltan/Repo/eov2etrs/geoid_eht2014.gtx' # geoid model for Z transformation
crs_eov = Proj(init='EPSG:23700', nadgrids=nadgrids, geoidgrids=geoidgrids) # HD72 / EOV
print(crs_eov.to_proj4())

# %% XYZ transformation
eov2etrs89_xyz = Transformer.from_proj(crs_eov, crs_etrs89_xyz)
p_etrs89_xyz = eov2etrs89_xyz.transform(y_eov, x_eov, z_eov)
print('Difference [cm]: ', np.linalg.norm(p_etrs89_xyz - p_eht_etrs89_xyz) * 100)

# %% LLH transformation:
eov2etrs89_llh = Transformer.from_proj(crs_eov, crs_etrs89_llh)
p_etrs89_llh = eov2etrs89_llh.transform(y_eov, x_eov, z_eov)
print('Difference [cm]: ', geodesic(p_etrs89_llh[0:2], p_eht_etrs89_llh[0:2]).m * 100)
print('Difference Z  [cm]: ', (p_etrs89_llh[2] - p_eht_etrs89_llh[2])*100)

# %% Inverse transform
etrs89_llh2eov = Transformer.from_proj(crs_etrs89_llh, crs_eov)
p_eov_hat = etrs89_llh2eov.transform(p_etrs89_llh[0], p_etrs89_llh[1], p_etrs89_llh[2])
print('Inverse transform: ', p_eov_hat)
print('Difference [cm]: ', np.linalg.norm(p_eov_hat - np.array([y_eov, x_eov, z_eov]))*100)

# %% Comparison to agt: http://www.agt.bme.hu/on_line/etrs2eov
p_agt_etrs89_llh = np.array([47.4809437, 19.0565298, 180.8083100])
print('Difference AGT-EHT [cm]: ', geodesic(p_agt_etrs89_llh[0:2], p_eht_etrs89_llh[0:2]).m * 100)
print('Difference AGT-Ours [cm]: ', geodesic(p_agt_etrs89_llh[0:2], p_etrs89_llh[0:2]).m * 100)
#print(geodesic(p_etrs89_llh, p_agt_etrs89_llh))

# %%
