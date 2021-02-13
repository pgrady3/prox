import numpy as np
import open3d as o3d
import sys
sys.path.append('/home/patrick/bed/SLP-Dataset-and-Code')
from data.SLP_RD import SLP_RD
import utils.utils as ut    # SLP utils
import cv2


def get_o3d_sphere(color=[0.3, 1.0, 0.3], pos=[0, 0, 0], radius=0.06):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=5)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color(color)
    mean = np.asarray(mesh_sphere.vertices).mean(axis=0)
    diff = np.asarray(pos) - mean
    mesh_sphere.translate(diff)
    return mesh_sphere


def set_camera_extrinsic(vis, transform=np.eye(4)):
    """
    :param vis: Open3D visualizer object
    :param transform: 4x4 numpy defining a rigid transform where the camera should go
    """

    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()
    cam.extrinsic = transform
    ctr.convert_from_pinhole_camera_parameters(cam)


def get_homography_components(h):
    '''((translationx, translationy), rotation, (scalex, scaley), shear)'''
    normalised_homography = h / h[2, 2]

    a = normalised_homography[0,0]
    b = normalised_homography[0,1]
    c = normalised_homography[0,2]
    d = normalised_homography[1,0]
    e = normalised_homography[1,1]
    f = normalised_homography[1,2]

    p = np.sqrt(a*a + b*b)
    r = (a*e - b*d)/p
    q = (a*d+b*e)/(a*e - b*d)

    translation = (c, f)
    scale = (p, r)
    shear = q
    theta = np.arctan2(b, a)

    return {'trans': translation, 'theta': theta, 'scale': scale, 'shear': shear}


def apply_homography(points, h):
    # Apply 3x3 homography matrix to points
    # THIS IS WRONG, HOMOGRAPHY SPECIFIED YX, NOT XY
    points_h = np.concatenate((points, np.ones(points.shape[0])))
    tform_h = np.matmul(h, points_h.T).T
    tform_h /= tform_h[:, 2]

    return tform_h[:, :2]


def henry_get_depth_homography(slp_dataset, idx):
    depth_Tr = slp_dataset.get_PTr_A2B(idx=idx, modA='depthRaw', modB='PM')     # Get SLP homography matrix
    depth_Tr /= depth_Tr[2, 2]  # Make more readable matrix

    # (192, 84) dimensions of pressure mat
    # (512, 424) dimensions of depth camera
    depth_Tr[0:2, 0:3] = depth_Tr[0:2, 0:3] / (192./345.)   # Scale matrix to align to PM. Is this a fudge factor? Bed is 345 depth pixels high?
    # print(get_homography_components(depth_Tr)
    # with np.printoptions(precision=4, suppress=True):
    #     print(depth_Tr)

    return depth_Tr


def henry_convert_depth_2_pc(slp_dataset, depth_arr, idx):
    depth_homography = henry_get_depth_homography(slp_dataset, idx)

    depth_arr_mod = cv2.warpPerspective(depth_arr, depth_homography, depth_arr.shape).astype(np.int16)  # Warp, and set output size
    depth_arr_mod[0, 0] = 2101  # Set magic point to fixed value, used for other stuff?

    cd_modified = np.matmul(depth_homography, np.array([slp_dataset.c_d[0], slp_dataset.c_d[1], 1.0]).T)    # Multiply the center of the depth image by homography
    cd_modified = cd_modified/cd_modified[2]    # Re-normalize

    ptc = ut.get_ptc(depth_arr_mod, slp_dataset.f_d, cd_modified[0:2], None) / 1000.0

    return ptc
