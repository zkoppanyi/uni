# %%
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, MultiPoint
import time

# %%
points_c_r_p = [[ -22.259937103098885 ,  -1.6843899580902115 ],
[ -7.700478230196843 ,  -2.070187709282951 ],
[ -6.959835753197708 ,  -0.6797709425023827 ],
[ -5.527861040598342 ,  -0.6793513714508965 ],
[ -4.935173816620718 ,  -1.9820968600726472 ],
[ -1.1988399976920434 ,  -2.106168271495848 ],
[ -0.448434010577527 ,  -0.7692861062517001 ],
[ 0.9450476234727576 ,  -0.8774028270124118 ],
[ 1.6390783928706063 ,  -2.1578976210069376 ],
[ 5.389512681483612 ,  -2.291609825306636 ],
[ 6.140628916759184 ,  -0.9939858096053334 ],
[ 7.603603076099733 ,  -0.9763755514348464 ],
[ 8.161596462797153 ,  -2.3232076548821237 ],
[ 11.92260547462389 ,  -2.4208310482450726 ],
[ 12.653568882004365 ,  -1.1363073523412013 ],
[ 13.97024082830039 ,  -1.0887166775792714 ],
[ 14.02643408745823 ,  4.1504856298613095 ],
[ 11.890876280685111 ,  4.045115063469355 ],
[ 5.266053837073811 ,  4.2110583232857275 ],
[ -1.294036075259115 ,  4.319972978864439 ],
[ -4.509165331816783 ,  4.430759891069019 ],
[ -22.515548070260795 ,  4.764583646358784 ],
[ -22.259937103098885 ,  -1.6843899580902115 ] ]

points_c_r_p = np.array(points_c_r_p)
poly = Polygon(points_c_r_p)

# %%
DISTANCE_BETWEEN_GENERATE_POINTS = 0.5
X_grid, Y_grid = np.meshgrid(np.arange(poly.bounds[0], poly.bounds[2], DISTANCE_BETWEEN_GENERATE_POINTS), np.arange(poly.bounds[1], poly.bounds[3], DISTANCE_BETWEEN_GENERATE_POINTS))
wall_points_c_r_p = np.array([X_grid.flatten(), Y_grid.flatten()]).T

# %%
plt.scatter(wall_points_c_r_p[:, 0], wall_points_c_r_p[:, 1], s=0.5)
poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, c='r')

# %%
def is_point_in_poly_fn(x, y):
    return poly.contains(Point((x, y)))

is_point_in_poly = np.vectorize(is_point_in_poly_fn)
start = time.time()
inside_poly_mask = is_point_in_poly(x=wall_points_c_r_p[:, 0], y=wall_points_c_r_p[:, 1])
wall_points_c_r_p_f = wall_points_c_r_p[inside_poly_mask, :]
end = time.time()
print("Vectorize : " + str((end-start))+ ' s')

# %%
plt.scatter(wall_points_c_r_p_f[:, 0], wall_points_c_r_p_f[:, 1], s=0.5)
poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, c='r')

# # %%
# for idx, (x, y) in enumerate(zip(poly.exterior.xy[0], poly.exterior.xy[1])):
#     d = np.sqrt(np.power(wall_points_c_r_p[:, 0] - x, 2) + np.power(wall_points_c_r_p[:, 1] - y, 2))

# %%
start = time.time()
# points = MultiPoint(wall_points_c_r_p)
points = [Point(wall_points_c_r_p[k, :]) for k in range(wall_points_c_r_p.shape[0])]
contained = np.fromiter(map(poly.contains, points), np.bool)
end = time.time()
print("Vectorize 2: " + str((end-start))+ ' s')

# %%
plt.scatter(wall_points_c_r_p[contained, 0], wall_points_c_r_p[contained, 1], s=0.5)
poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, c='r')

# %% Inscribed circle
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(np.array([poly_x, poly_y]).T)
# fig = voronoi_plot_2d(vor)
# plt.show()
max_dist, max_ind = -np.inf, 0
for k in range(len(vor.vertices)):
    pt = Point(vor.vertices[k, 0], vor.vertices[k, 1])
    if poly.contains(pt):
        d = poly.exterior.distance(pt)
        if d > max_dist:
            max_dist, max_ind = d, k

center = vor.vertices[max_ind, :]
radius = max_dist

print(max_dist)
print(max_ind)

# %% Plot inscirbed cirlc plot
plt.scatter(wall_points_c_r_p[:, 0], wall_points_c_r_p[:, 1], s=0.5)
poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, c='r')
t = np.arange(0, 2*np.pi, 0.01)
plt.plot(center[0] + radius*np.cos(t), center[1] + radius*np.sin(t))
plt.axis('equal')

# %%
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
# def constraints(params):
#     x, y, w, h = params[0], params[1], params[2], params[3]
#     rect = Polygon([[x, y], [x+h, y], [x+h, y+w], [x, y+w], [x, y]])
#     return poly.intersection(rect).area - w*h

def constraints(params):
    x, y, w, h = params[0], params[1], params[2], params[3]
    return [int(poly.contains(Point(x, y))),
            int(poly.contains(Point(x+w, y))),
            int(poly.contains(Point(x+w, y+h))),
            int(poly.contains(Point(x, y+h)))]

def cost_fn(params):
    return -1 * params[2] * params[3]

def points_inside_exterior_constraint(params):
    x, y, w, h = params[0], params[1], params[2], params[3]
    return [x, y, x+w, y+h]

linear_constraint_fn = NonlinearConstraint(points_inside_exterior_constraint,
                        [poly.bounds[0], poly.bounds[1], poly.bounds[0], poly.bounds[1]],
                        [poly.bounds[2], poly.bounds[3], poly.bounds[2], poly.bounds[3]])

nonlinear_constraint_fn = NonlinearConstraint(constraints, [1, 1, 1, 1], [1, 1, 1, 1])

res = minimize(cost_fn, [0, 0, 10, 10], method='trust-constr',
               constraints=[nonlinear_constraint_fn],
               options={'verbose': 1})

# %%
plt.scatter(wall_points_c_r_p[:, 0], wall_points_c_r_p[:, 1], s=0.5)
poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, c='r')
x, y, w, h = res.x[0], res.x[1], res.x[2], res.x[3]
plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], c='g')

# %%
# Try to do something like this:
# https://github.com/planetlabs/maxrect/blob/c91c240279cab57e77d348010c2683f944c653db/maxrect/__init__.py#L93
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

def two_pts_to_line(pt1, pt2):
    """
    Create a line from two points in form of
    a1(x) + a2(y) = b
    """
    pt1 = [float(p) for p in pt1]
    pt2 = [float(p) for p in pt2]
    try:
        slp = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    except ZeroDivisionError:
        slp = 1e5 * (pt2[1] - pt1[1])
    a1 = -slp
    a2 = 1.
    b = -slp * pt1[0] + pt1[1]

    return a1, a2, b


def pts_to_leq(coords):
    """
    Converts a set of points to form Ax = b, but since
    x is of length 2 this is like A1(x1) + A2(x2) = B.
    returns A1, A2, B
    """

    A1 = []
    A2 = []
    B = []
    rangeX = []
    rangeY = []
    for i in range(len(coords) - 1):
        pt1 = coords[i, :]
        pt2 = coords[i + 1, :]
        a1, a2, b = two_pts_to_line(pt1, pt2)
        A1.append(a1)
        A2.append(a2)
        B.append(b)
        if pt2[0] > pt1[0]:
            rangeX.append([pt1[0], pt2[0]])
        else:
            rangeX.append([pt2[0], pt1[0]])

        if pt2[1] > pt1[1]:
            rangeY.append([pt1[1], pt2[1]])
        else:
            rangeY.append([pt2[1], pt1[1]])

        rangeY.append([pt1[1], pt2[1]])
    return A1, A2, B, np.array(rangeX), np.array(rangeY)

inside_pt = (poly.representative_point().x, poly.representative_point().y)

poly_x, poly_y = poly.exterior.xy
sc_coordinates = np.array([poly_x, poly_y]).T

A1, A2, B, rangeX, rangeY = pts_to_leq(sc_coordinates)

upper_bounds = []
lower_bounds = []
BIG_NUMBER = 1000000
for i in range(len(B)):
    # if inside_pt[0] * A1[i] + inside_pt[1] * A2[i] <= B[i]:
    #     upper_bounds.extend([B[i], B[i], B[i], B[i]])
    #     lower_bounds.extend([-BIG_NUMBER, -BIG_NUMBER, -BIG_NUMBER, -BIG_NUMBER])
    # else:
    #     upper_bounds.extend([BIG_NUMBER, BIG_NUMBER, BIG_NUMBER, BIG_NUMBER])
    #     lower_bounds.extend([B[i], B[i], B[i], B[i]])
    upper_bounds.extend([BIG_NUMBER, BIG_NUMBER, BIG_NUMBER, BIG_NUMBER])
    lower_bounds.extend([0, 0, 0, 0])
upper_bounds = np.array(upper_bounds)
lower_bounds = np.array(lower_bounds)

def fn_constraints(params):
    x, y, w, h = params[0], params[1], params[2], params[3]
    bl = [x, y]
    tl = [x, y+h]
    br = [x+w, y]
    tr = [x+w, y+h]
    cons = []
    for i in range(len(B)):
        cons_line = [0, 0, 0, 0]
        b = -B[i]
        s = 1
        if inside_pt[0] * A1[i] + inside_pt[1] * A2[i] >= B[i]:
            b = -b
            s = -1

        if rangeX[i, 0] < bl[0] and bl[0] < rangeX[i, 1] and rangeY[i, 0] < bl[1] and bl[1] < rangeY[i, 1]:
            cons_line[0] = s*(bl[0] * A1[i] + bl[1] * A2[i] + b)
        else:
            cons_line[0] = s*(bl[0] * A1[i] + bl[1] * A2[i] + b)

        if rangeX[i, 0] < tr[0] and tr[0] < rangeX[i, 1] and rangeY[i, 0] < tr[1] and tr[1] < rangeY[i, 1]:
            cons_line[1] = s*(tr[0] * A1[i] + tr[1] * A2[i] + b)
        else:
            cons_line[1] = s*(tr[0] * A1[i] + tr[1] * A2[i] + b)

        if rangeX[i, 0] < br[0] and br[0] < rangeX[i, 1] and rangeY[i, 0] < br[1] and br[1] < rangeY[i, 1]:
            cons_line[2] = s*(br[0] * A1[i] + br[1] * A2[i] + b)
        else:
           cons_line[2] = s*(br[0] * A1[i] + br[1] * A2[i] + b)

        if rangeX[i, 0] < tl[0] and tl[0] < rangeX[i, 1] and rangeY[i, 0] < tl[1] and tl[1] < rangeY[i, 1]:
            cons_line[3] = s*(tl[0] * A1[i] + tl[1] * A2[i] + b)
        else:
           cons_line[3] = s*(tl[0] * A1[i] + tl[1] * A2[i] + b)

        cons.extend(cons_line)

    return np.array(cons)

linear_constraint_fn = NonlinearConstraint(fn_constraints, lower_bounds, upper_bounds)

def cost_fn(params):
    return -1 * params[2] * params[3]

res = minimize(cost_fn, [0, 0, 10, 10], method='trust-constr',
               constraints=[linear_constraint_fn],
               options={'verbose': 1})

# %%
plt.scatter(wall_points_c_r_p[:, 0], wall_points_c_r_p[:, 1], s=0.5)
poly_x, poly_y = poly.exterior.xy
plt.plot(poly_x, poly_y, c='r')
x, y, w, h = res.x[0], res.x[1], res.x[2], res.x[3]
plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], c='g')
plt.scatter(inside_pt[0], inside_pt[1], c='r')

# %%
poly_x, poly_y = poly.exterior.xy
max_rect = None
max_area = -np.inf
#plt.plot(poly.exterior.xy[0], poly.exterior.xy[1], c='r')
for k1 in range(len(poly_x)):
    for k2 in range(len(poly_x)):
        x, y = poly_x[k1], poly_y[k1]
        w = poly_x[k2] - x
        h = poly_y[k2] - y
        rect = Polygon([[x, y], [x+w, y], [x+w, y+h], [x, y+h], [x, y]])
        area = np.abs(w*h)

        #plt.plot(rect.exterior.xy[0], rect.exterior.xy[1], c='b')

        if poly.contains(rect) and area > max_area:
            max_area = area
            max_rect = rect

# %%
plt.scatter(wall_points_c_r_p[:, 0], wall_points_c_r_p[:, 1], s=0.5)
poly_x, poly_y = poly.exterior.xy
plt.plot(poly.exterior.xy[0], poly.exterior.xy[1], c='r')
plt.plot(max_rect.exterior.xy[0], max_rect.exterior.xy[1], c='b')

# %%
