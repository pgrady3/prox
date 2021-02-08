import numpy as np
import open3d as o3d
import sys
sys.path.append('/home/patrick/bed/SLP-Dataset-and-Code')
from data.SLP_RD import SLP_RD
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


def henry_convert_depth_2_pc(slp_dataset, depth_arr):
    c_d = [208.1, 259.7]  # z/f = x_m/x_p so m or mm doesn't matter
    f_d = [367.8, 367.8]

    calibrate_depth = True

    if calibrate_depth:
        depth_Tr = slp_dataset.genPTr_dict(['depth'])['depth'][0] #self.get_PTr_A2B(modA='depth', modB='PM')
        # print(self.get_PTr_A2B(modA='depth', modB='PM'))
        depth_Tr[0:2, 0:3] = depth_Tr[0:2, 0:3]/ (192./345.)

        # print('depthTR', depth_Tr)
        # print(np.matmul(depth_Tr, np.array([self.c_d[0], self.c_d[1], 1.0]).T))
        cd_modified = np.matmul(depth_Tr, np.array([c_d[0], c_d[1], 1.0]).T)
        cd_modified = cd_modified/cd_modified[2]

        depth_arr_mod = cv2.warpPerspective(depth_arr, depth_Tr, tuple([155, 345])).astype(np.int16)  #size of depth arr input is
        depth_arr_mod[0,0] = 2101

        ptc = slp_dataset.get_ptc(depth_arr_mod, f_d, cd_modified[0:2], None)/1000
        # ptc = self.get_ptc(depth_arr_mod, self.f_d, self.c_d, None)/1000

    else:
        ptc = slp_dataset.get_ptc(depth_arr, f_d, c_d, None)/1000

    filter_pc = False
    if filter_pc:
        rot_angle_fixed = np.deg2rad(3.0)
        # rot_angle_fixed = np.deg2rad(10.0)
        ptc[:, 0] = (ptc[:, 0])*np.cos(rot_angle_fixed) - (ptc[:, 2])*np.sin(rot_angle_fixed)
        ptc[:, 2] = (ptc[:, 0])*np.sin(rot_angle_fixed) + (ptc[:, 2])*np.cos(rot_angle_fixed)

        rot_angle = np.deg2rad(2.5) #
        ptc[:, 1] = (ptc[:, 1])*np.cos(rot_angle) - (ptc[:, 2])*np.sin(rot_angle)
        ptc[:, 2] = (ptc[:, 1])*np.sin(rot_angle) + (ptc[:, 2])*np.cos(rot_angle)

        ptc = ptc[ptc[:, 2] < 2.103]

        ptc[:, 1] = (ptc[:, 1])*np.cos(-rot_angle) - (ptc[:, 2])*np.sin(-rot_angle)
        ptc[:, 2] = (ptc[:, 1])*np.sin(-rot_angle) + (ptc[:, 2])*np.cos(-rot_angle)

    ptc_first_point = np.array(ptc[0])

    length_new_pmat = 1.92
    width_new_pmat = 0.84

    scale_diff_h = (length_new_pmat - 64*0.0286)
    scale_diff_w = (width_new_pmat - 27*0.0286)

    # this is because we set the first point in the depth image to 2101.
    ptc[:, 0] -= ptc_first_point[0]
    ptc[:, 1] -= ptc_first_point[1]
    ptc[:, 2] -= ptc_first_point[2]

    ptc[:, 0] -= (scale_diff_w)
    ptc[:, 1] -= (length_new_pmat - scale_diff_h)

    ptc = np.concatenate((-ptc[:, 1:2], ptc[:, 0:1], ptc[:, 2:3]), axis = 1)

    if filter_pc:
        ptc = ptc[ptc[:, 1] > -0.01, :] #cut off points at the edge
        ptc = ptc[ptc[:, 2] > -0.5, :] #cut off stuff thats way up high
        ptc = ptc[ptc[:, 2] < 0.01]

    return ptc