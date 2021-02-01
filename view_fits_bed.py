import sys
# Set this to the SLP github repo
sys.path.append('/home/patrick/bed/SLP-Dataset-and-Code')
import numpy as np
import open3d as o3d
import os
from data.SLP_RD import SLP_RD
import json
import utils.utils as ut    # SLP utils
import torch
from tqdm import tqdm
import pickle
import smplx
import trimesh
from prox.misc_utils import text_3d
import matplotlib.cm as cm
from prox.camera import PerspectiveCamera
from prox.misc_utils import smpl_to_openpose
import pprint
import argparse
from patrick_util import *


SLP_PATH = '/home/patrick/datasets/SLP/danaLab'
FITS_PATH = '/home/patrick/bed/prox/slp_fits'

def get_smpl(pkl_data, json_data):
    gender = json_data['people'][0]['gender_gt']
    print('Target height {}, weight {}'.format(json_data['people'][0]['height'], json_data['people'][0]['weight']))

    betas = torch.Tensor(pkl_data['betas']).unsqueeze(0)
    print('Betas:', pkl_data['betas'])
    pose = torch.Tensor(pkl_data['body_pose']).unsqueeze(0)
    transl = torch.Tensor(pkl_data['transl']).unsqueeze(0)
    global_orient = torch.Tensor(pkl_data['global_orient']).unsqueeze(0)

    model = smplx.create('models', model_type='smpl', gender=gender)
    output = model(betas=betas, body_pose=pose, transl=transl, global_orient=global_orient, return_verts=True)
    smpl_vertices = output.vertices.detach().cpu().numpy().squeeze()
    smpl_joints = output.joints.detach().cpu().numpy().squeeze()

    output_unposed = model(betas=betas, body_pose=pose * 0, transl=transl, global_orient=global_orient, return_verts=True)
    smpl_vertices_unposed = output_unposed.vertices.detach().cpu().numpy().squeeze()

    for i, lbl in enumerate(['Wingspan', 'Height', 'Thickness']):
        print('Actual', lbl, smpl_vertices_unposed[:, i].max() - smpl_vertices_unposed[:, i].min(), end=' ')
    print()

    smpl_trimesh = trimesh.Trimesh(vertices=np.asarray(smpl_vertices_unposed), faces=model.faces)
    print('Est weight from volume', smpl_trimesh.volume * 1.03 * 1000)
    # print('Pose embedding', pkl_data['pose_embedding'])
    # print('Body pose', np.array2string(pkl_data['body_pose'], separator=', '))
    # print('SMPL vertex zero', smpl_vertices[0, :])

    smpl_o3d = o3d.geometry.TriangleMesh()
    smpl_o3d.triangles = o3d.utility.Vector3iVector(model.faces)
    smpl_o3d.vertices = o3d.utility.Vector3dVector(smpl_vertices)
    smpl_o3d.compute_vertex_normals()
    # smpl_o3d.paint_uniform_color([0.3, 0.3, 0.3])

    smpl_o3d_2 = o3d.geometry.TriangleMesh()
    smpl_o3d_2.triangles = o3d.utility.Vector3iVector(model.faces)
    smpl_o3d_2.vertices = o3d.utility.Vector3dVector(smpl_vertices + np.array([1.3, 0, 0]))
    smpl_o3d_2.compute_vertex_normals()
    smpl_o3d_2.paint_uniform_color([0.7, 0.3, 0.3])

    # Visualize SMPL joints - Patrick

    camera = PerspectiveCamera(rotation=torch.tensor(pkl_data['camera_rotation']).unsqueeze(0),
                               translation=torch.tensor(pkl_data['camera_translation']).unsqueeze(0),
                               center=torch.tensor(pkl_data['camera_center']).unsqueeze(0),
                               focal_length_x=torch.tensor(pkl_data['camera_focal_length_x']),
                               focal_length_y=torch.tensor(pkl_data['camera_focal_length_y']))

    all_markers = []
    # all_markers.append(get_o3d_sphere(pos=pkl_data['camera_translation']))    # Add dot where camera should be, but it screws up point cloud coloring
    for i in range(25):
        if np.all(pkl_data['gt_joints'][i, :] == 0):
            continue

        cmap_val = (i / 25.0 * 3) % 1
        color = cm.jet(cmap_val)[:3]

        pos = smpl_joints[smpl_to_openpose('smpl')[i], :]
        rad = 0.07
        if i == 0:
            ear_joints = smpl_to_openpose('smpl')[17:19]    # Get left and right ear
            pos = smpl_joints[ear_joints, :].mean(0)
            rad = 0.10

        smpl_marker = get_o3d_sphere(color=color, pos=pos, radius=rad)
        all_markers.append(smpl_marker)

        z_depth = smpl_joints[smpl_to_openpose('smpl')[i], 2] - 0.20
        gt_pos_3d = camera.inverse_camera_tform(torch.tensor(pkl_data['gt_joints']).unsqueeze(0), z_depth).detach().squeeze(0).cpu().numpy()

        pred_marker = get_o3d_sphere(color=color, pos=gt_pos_3d[i, :], radius=0.03)
        all_markers.append(pred_marker)

    return smpl_vertices, model.faces, smpl_o3d, smpl_o3d_2, all_markers


def apply_homography(points, h):
    # Apply 3x3 homography matrix to points
    points_h = np.concatenate((points, np.ones(points.shape[0])))
    tform_h = np.matmul(h, points_h.T).T
    tform_h /= tform_h[:, 2]

    return tform_h[:, :2]


def get_depth(idx, sample):
    # arr_IR2depth = SLP_dataset.get_array_A2B(idx=idx, modA='IR', modB='depthRaw')
    # arr_PM2depth = SLP_dataset.get_array_A2B(idx=idx, modA='PM', modB='depthRaw')
    pressure_to_depth_homography = SLP_dataset.get_PTr_A2B(idx=idx, modA='PM', modB='depthRaw')     # Get homography matrix from A to B
    depth_to_depth_homography = SLP_dataset.get_PTr_A2B(idx=idx, modA='depthRaw', modB='depth')     # Get homography matrix from A to B

    depth, jt, bb = SLP_dataset.get_array_joints(idx_smpl=idx, mod='depthRaw', if_sq_bb=False)
    bb = bb.round().astype(int)
    bb += np.array([-25, -5, 50, 10])    # Patrick, expand "bounding box", since it often cuts off parts of the body
    pointcloud = ut.get_ptc(depth, SLP_dataset.f_d, SLP_dataset.c_d, bb) / 1000.0

    valid_pcd = np.logical_and(pointcloud[:, 2] > 1.55, pointcloud[:, 2] < 2.15)  # Cut out any outliers above the bed
    pointcloud = pointcloud[valid_pcd, :]
    ptc_depth = o3d.geometry.PointCloud()
    ptc_depth.points = o3d.utility.Vector3dVector(pointcloud)
    return ptc_depth


def get_rgb(sample):
    # Load RGB image
    rgb_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'images', 'image_{:06d}'.format(sample[2]), '000', 'output.png')
    rgb_image = o3d.io.read_image(rgb_path)
    rgb_raw = np.asarray(rgb_image)
    depth_raw = np.ones((rgb_raw.shape[0], rgb_raw.shape[1]), dtype=np.float32) * 2.15
    depth_image = o3d.geometry.Image(depth_raw)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_scale=1)

    f_r = [902.6, 877.4]    # From SLP dataset README
    c_r = [278.4, 525.1]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=rgb_raw.shape[1], height=rgb_raw.shape[0],
                                                  fx=f_r[0], fy=f_r[1], cx=c_r[0], cy=c_r[1])

    rgbd_ptc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2ColorCameraDefault)

    return rgbd_ptc


def view_fit(sample, idx):
    # if sample[0] < 4 or sample[2] < 43:
    #     return

    pkl_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'results', 'image_{:06d}'.format(sample[2]), '000.pkl')
    if not os.path.exists(pkl_path):
        return

    print('Reading', pkl_path)
    pkl_np = pickle.load(open(pkl_path, 'rb'))

    json_path = os.path.join(FITS_PATH, 'keypoints', '{}_{:05d}'.format(sample[1], sample[0]), 'image_{:06d}_keypoints.json'.format(sample[2]))
    with open(json_path) as keypoint_file:
        json_data = json.load(keypoint_file)

    smpl_vertices, smpl_faces, smpl_mesh, smpl_mesh_calc, joint_markers = get_smpl(pkl_np, json_data)
    pcd = get_depth(idx, sample)
    rgbd_ptc = get_rgb(sample)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(smpl_mesh)
    vis.add_geometry(smpl_mesh_calc)
    vis.add_geometry(rgbd_ptc)
    lbl = 'Participant {} sample {}'.format(sample[0], sample[2])
    vis.add_geometry(text_3d(lbl, (-0.5, 1.0, 2), direction=(0.01, 0, -1), degree=-90, font_size=150, density=0.2))

    for j in joint_markers:
        vis.add_geometry(j)

    set_camera_extrinsic(vis, np.eye(4))
    vis.run()
    vis.destroy_window()
    print('\n')


def make_dataset(skip_sample=0, skip_participant=0):
    all_samples = SLP_dataset.pthDesc_li

    for idx, sample in enumerate(tqdm(all_samples)):
        if sample[0] < skip_participant or sample[2] < skip_sample:
            continue

        view_fit(sample, idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sample', type=int, default=0)
    parser.add_argument('-p', '--participant', type=int, default=0)
    args = parser.parse_args()

    class PseudoOpts:
        SLP_fd = SLP_PATH
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover']  # give the cover class you want here
    SLP_dataset = SLP_RD(PseudoOpts, phase='all')  # all test result

    make_dataset(skip_sample=args.sample, skip_participant=args.participant)
