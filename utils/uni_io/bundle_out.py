import numpy as np
import math

PINHOLE_CAMERA_RADIAL3 = 3
PINHOLE_CAMERA_BROWN = 4
BROWN_CAMERA_INTRINSICS = 8

class Cameras(object):
    def __init__(self, cam_info=0):
        self.type = 1

def opk_M(M):
    '''
    Extracts the tertiary rotation angles from the Rotation matrix
    '''
    o = math.atan2(-M[2][1], M[2][2])
    p = math.atan2(M[2][0], np.sqrt(M[2][1]**2 + M[2][2]**2))
    k = math.atan2(-M[1][0], M[0][0])
    return [o,p,k]

def read_bundle_out(bundle_file):
    '''
    parses the input bundle file and creates a camera class
    '''
    # reads the the bundle file
    cam_file = open(bundle_file, "r+")  #this works!
    #cam_file = open("bundle_free.out", "r+")  #this works!
    line1 = cam_file.readline() #reads header
    line2 = cam_file.readline()

    str2='num_cameras='
    dlin=line2.find(str2)  #delineate cameras
    temp_string = line2[dlin:]
    testy = temp_string.split('=')
    n_ground_pts = int(testy[2])

    # create camera object and populate
    cam=Cameras(cam_file)

    cam.n_cams= int(testy[1].split(' ')[0])
    cam.f={}
    cam.k1={};cam.k2={};cam.k3 = {}
    cam.name = {}
    cam.size_x = {}; cam.size_y = {}
    cam.xo = {}; cam.yo={}
    cam.means = {}
    cam.t1 = {}; cam.t2 = {}
    cam.camera_model =  PINHOLE_CAMERA_RADIAL3
    # # throw away means
    line2 = cam_file.readline()
    all_means = line2.split()
    x_mean = float(all_means[1].split('=')[1])
    y_mean = float(all_means[2].split('=')[1])
    z_mean = float(all_means[3].split('=')[1])
    cam.means[0] = x_mean; cam.means[1] = y_mean; cam.means[2] = z_mean

    # initialize camera rotation and translation
    R=np.zeros((3,3,cam.n_cams))
    tran=np.zeros((cam.n_cams,3))
    cam.pos=np.zeros((cam.n_cams,3))
    cam.pos_v=np.zeros((cam.n_cams,3))
    cam.opk=np.zeros((cam.n_cams,3))

    # loop 4 rows/cam * n_cams
    q=0

    #for line in cam_file:
    while q<= int(cam.n_cams)-1:
        # get f, k1, k2  (interior orientation output from bundle)
        cam.name[q] = cam_file.readline().split()
        #throw away line
        dims = cam_file.readline()
        width, height = dims.split()
        cam.size_x[q] = width
        cam.size_y[q] = height

        fkk=cam_file.readline().split()
        cam.f[q]=float(fkk[0])
        cam.k1[q] = float(fkk[2])
        cam.k2[q] = float(fkk[3])
        cam.k3[q] = float(fkk[4])
        cam.xo[q] = float(fkk[5])
        cam.yo[q] = float(fkk[6])
        cam.pix_width = float(fkk[1])
        if len(fkk)==9:
            cam.camera_model =  PINHOLE_CAMERA_BROWN
            cam.t1[q] = float(fkk[7])
            cam.t2[q] = float(fkk[8])
        # get R
        for r in range(0,3):
            R[r,:,q]=cam_file.readline().split()

        # get translation (t)
        tran[q,:]=cam_file.readline().split()

        #camera position in bundler coord. system
        cam.pos[q,:]=np.dot(-R[:,:,q].T,tran[q,:])

        #camera viewing direction in bundler coord. system
        cam.pos_v[q,:]=np.dot(np.transpose(-R[:,:,q]),np.transpose([0,0,-1]))

        #throw away opk line
        cam.opk[q,:] = opk_M(R[:,:,q])
        opk = cam_file.readline().split()
        q=q+1

    img_X=cam.pos
    g_XYZ=[]

    #now, read in ground points from bundle.out file and parse the relevant information
    g_xyz=np.zeros([n_ground_pts,3])
    color_xyz=np.zeros([n_ground_pts,3])
    g_xyz_data={}
    for i in range(int(n_ground_pts)):
        g_xyz[i,:]=cam_file.readline().split()
        color_xyz[i,:]=cam_file.readline().split()
        g_xyz_data[i]=cam_file.readline().split()

    x=g_xyz[:,0]
    y=g_xyz[:,1]
    z=g_xyz[:,2]

    # first, lets parse the keypoint information from g_xyz_data

    #initialize list specifying total # of cameras that view i-th point
    n_view_pt=np.zeros([n_ground_pts,1])

    #initialize list that contains which cameras view i-th point
    cam_view_pt=[]

    #initalize list that contains x and y pixel location for each camera that views i-th point
    im_x=[]
    im_y=[]
    row = []
    col = []
    row2 = []
    col2 = []
    im_x_c = []
    im_y_c = []
    xo = cam.xo[0]
    yo = cam.yo[0]
    k1 = cam.k1[0]
    k2 = cam.k2[0]
    k3 = cam.k3[0]
    ##### to view the first point in bundle file in the original distorted images
    ##### start
    err = []
    for i in range(len(g_xyz_data)):

        n_view_pt[i]=int(g_xyz_data[i][0])
        #append for each i-th point
        cam_view_pt.append(g_xyz_data[i][1::4])
        im_x.append(g_xyz_data[i][3::4])
        im_y.append(g_xyz_data[i][4::4])

    return cam, R, tran, [x,y,z], [x_mean, y_mean, z_mean], cam_view_pt, n_view_pt