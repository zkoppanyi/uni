# %%
import numpy as np
from pyproj import CRS, Transformer, Proj
from geopy.distance import geodesic

# %%
crs_eov = CRS.from_epsg(32629)
crs_etrs89_llh = CRS.from_epsg(4326)

print(crs_eov.to_wkt(pretty=True))

# %%
eov2etrs89_llh = Transformer.from_crs(crs_eov, crs_etrs89_llh)

# %%
coors = [[473609.371,4289134.097,220.438],
[473616.690,4289124.178,219.960],
[473630.007,4289104.414,219.897]]

for coor in coors:
    llh = eov2etrs89_llh.transform(coor[0], coor[1], coor[2])
    print(llh[0], llh[1], llh[2])

# %%
