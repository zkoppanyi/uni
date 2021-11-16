# %% External imports
#%matplotlib qt

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import viz

# %%
## Plot a single camera camera fustrum

# %%
X = np.array([0, 0, 0])
R = np.eye(3)
f = 2
scale = 1

# %%
fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
viz.plot_fustrum(ax, X, R, f=f, scale=scale)
viz.set_3d_axes_equal(ax)

# %%
## Plot multiple camera fustrums

# You might use the following code to generate data from C++
# std::cout << "R_cams.append(";
# std::cout << "np.array([[" << R(0, 0) << "," << R(0, 1) << "," << R(0, 2) << "],";
# std::cout << "[" << R(1, 0) << "," << R(1, 1) << "," << R(1, 2) << "],";
# std::cout << "[" << R(2, 0) << "," << R(2, 1) << "," << R(2, 2) << "]])";
# std::cout << ")" << std::endl;
# std::cout << "t_cams.append(";
# std::cout << "np.array([" << t(0) << "," << t(1) << "," << t(2) << "])";
# std::cout << ")" << std::endl;

# %% Pose definiations
R_cams = []
t_cams = []

R_cams.append(np.array([[0.46224950999999997,-0.88642706999999998,0.023925829999999999],[-0.88672448999999998,-0.46227224,0.0049042499999999998],[0.0067129900000000003,-0.023482610000000001,-0.99970170999999997]]))
t_cams.append(np.array([37.288039423382166,-10.165071594832114,1.8727184629264104]))
R_cams.append(np.array([[0.46651118000000003,-0.88420001999999998,0.02361473],[-0.88449507999999999,-0.46651474999999998,0.0056951700000000003],[0.0059809499999999996,-0.023543979999999999,-0.99970490999999995]]))
t_cams.append(np.array([37.659737128609429,6.2026050253511551,1.806322197411959]))
R_cams.append(np.array([[0.46728996,-0.88381131000000002,0.02275201],[-0.88407325000000003,-0.46733393000000001,0.0036717999999999998],[0.0073876100000000002,-0.021830229999999999,-0.99973440000000002]]))
t_cams.append(np.array([37.110483017656314,-26.062842185291906,2.1648932375751877]))
R_cams.append(np.array([[0.49233894,-0.87008560000000001,0.023524920000000001],[-0.87039502000000002,-0.49227402999999997,0.0088764099999999995],[0.00385747,-0.024846179999999999,-0.99968383999999999]]))
t_cams.append(np.array([-5.4967235887385559,31.091132695336771,1.610341169355259]))
R_cams.append(np.array([[0.50085186000000004,-0.86521892,0.02331571],[-0.86552704000000003,-0.50076681999999995,0.0097747300000000006],[0.0032184499999999999,-0.025076060000000001,-0.99968036999999998]]))
t_cams.append(np.array([-5.5377931849465023,14.710846119800054,1.7300687232930638]))
R_cams.append(np.array([[0.47180678999999998,-0.88143651000000001,0.021633989999999999],[-0.88162320000000005,-0.47195058000000001,-0.0017871199999999999],[0.01178541,-0.018229849999999999,-0.99976436000000002]]))
t_cams.append(np.array([37.384051130187636,-41.633462546113947,2.1158634448314526]))
R_cams.append(np.array([[0.47189439999999999,-0.88132094000000005,0.024270690000000001],[-0.88162231000000002,-0.47193537000000002,0.0043717900000000004],[0.0076012500000000004,-0.023460600000000002,-0.99969585999999999]]))
t_cams.append(np.array([37.970831512826429,22.702039449830167,2.2784408494325161]))
R_cams.append(np.array([[0.49323628000000003,-0.86958290999999999,0.023313810000000001],[-0.86987912999999994,-0.49321372000000002,0.0071084499999999997],[0.0053173099999999996,-0.02378634,-0.99970292000000005]]))
t_cams.append(np.array([-5.7895992643088299,47.41667398960805,1.3434057619016124]))
R_cams.append(np.array([[0.52530100000000002,-0.85062040000000005,0.022445380000000001],[-0.85089974999999995,-0.52527330999999999,0.0075872400000000003],[0.0053360999999999999,-0.02308435,-0.99971927999999999]]))
t_cams.append(np.array([-4.180344666992224,-17.63113751842792,1.6293067869202613]))
R_cams.append(np.array([[0.67739077000000003,-0.73556431,0.0093213199999999993],[-0.73555954000000001,-0.67744453999999998,-0.0045894100000000004],[0.0096904799999999996,-0.00374756,-0.99994601999999999]]))
t_cams.append(np.array([-54.873649641752209,13.564125668214011,2.0621008471261026]))
R_cams.append(np.array([[0.46154272000000002,-0.88673528000000001,0.026055149999999999],[-0.88708531000000002,-0.46157870000000001,0.0049759699999999997],[0.0076141400000000001,-0.02540976,-0.99964812000000003]]))
t_cams.append(np.array([38.403466611833693,38.517046653437177,2.6255313556474733]))
R_cams.append(np.array([[0.52497444000000004,-0.85077981999999996,0.023990129999999998],[-0.85109480000000004,-0.52495965,0.00741715],[0.0062834900000000001,-0.02431169,-0.99968467999999999]]))
t_cams.append(np.array([-3.1792848847595057,-33.939470542312002,1.3347059664748238]))
R_cams.append(np.array([[0.68604211999999998,-0.72722567999999999,0.02211378],[-0.72739374999999995,-0.68622002999999998,-0.00063688000000000004],[0.015638079999999999,-0.015648499999999999,-0.99975526000000003]]))
t_cams.append(np.array([-49.141419809434737,-2.7622727471745492,1.8123658774813083]))
R_cams.append(np.array([[0.66895943999999996,-0.74293441999999998,0.023274739999999999],[-0.74317641000000001,-0.66908968000000002,0.0027978299999999998],[0.013494280000000001,-0.019168879999999999,-0.99972519000000004]]))
t_cams.append(np.array([-45.14375735420402,-17.020471172497622,0.74002029836796157]))
R_cams.append(np.array([[0.46577089999999999,-0.88465850000000001,0.020900169999999999],[-0.88480402999999996,-0.46594503999999998,-0.0041277900000000001],[0.013390020000000001,-0.01656995,-0.99977305000000005]]))
t_cams.append(np.array([36.921765921296576,-58.333465843194972,2.0277098339924837]))
R_cams.append(np.array([[0.48385872000000002,-0.87480838999999999,0.02431088],[-0.87512471000000003,-0.48385447999999998,0.0064485300000000001],[0.0061216999999999999,-0.02439523,-0.99968365000000003]]))
t_cams.append(np.array([-5.3925073936428083,63.572360842694366,1.2795696460166441]))
R_cams.append(np.array([[0.64499671000000003,-0.76379189000000003,0.024519200000000001],[-0.76410549000000005,-0.64506109,0.0062437899999999999],[0.011047420000000001,-0.02276247,-0.99967985999999998]]))
t_cams.append(np.array([-42.318382286406617,-31.469204848937117,0.40737715271141517]))
R_cams.append(np.array([[0.61809161000000001,-0.78570704000000002,0.025044219999999999],[-0.78604348999999996,-0.61813068000000004,0.0070777100000000001],[0.0099196000000000006,-0.024060519999999998,-0.99966129000000004]]))
t_cams.append(np.array([-40.624343186101939,-45.960225675511538,0.35988156190389997]))

pts = np.array([[44.5, -22.09, -78.29]])
# %%
fig = plt.figure(figsize=(12,10))
ax = plt.axes(projection='3d')
ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], c='g', marker='o')
for k in range(len(R_cams)):
    X = -R_cams[k].T @ t_cams[k]
    viz.plot_fustrum(ax, X, R_cams[k], f=f, scale=10)
viz.set_3d_axes_equal(ax)

# %%
