import sys
sys.path.append('/home/patrick/bed/SLP-Dataset-and-Code')
import numpy as np
import open3d as o3d
import os
from glob import glob
from shutil import copyfile
from data.SLP_RD import SLP_RD
import json
import cv2
import utils.utils as ut    # SLP utils
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from prox.camera import PerspectiveCamera
import torch
from tqdm import tqdm
from patrick_util import *


SLP_PATH = '/home/patrick/datasets/SLP/danaLab'
OUT_PATH = '/home/patrick/bed/prox/slp_tform'
phases = ['uncover']

WARP_TO_PM = True
BED_INLIER_THRESH = -0.02


def phyvec_to_dict(phy_vec):
    # From SLP docs: weight, height, gender(1:male, 0:female), bust, waist, hip， right upper arm， right lower arm， right upper leg， right lower leg
    d = dict()
    d['weight'] = phy_vec[0]
    d['height'] = phy_vec[1]
    d['gender_gt'] = 'male'
    if phy_vec[2] < 0.5:
        d['gender_gt'] = 'female'

    return d


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def copy_images(sample, idx):
#     dest_folder = os.path.join(OUT_PATH, 'recordings', '{}_{:05d}'.format(sample[1], sample[0]))
#     dest_color = os.path.join(dest_folder, 'Color')
#     dest_filename = os.path.join(dest_color, 'image_{:06d}.png'.format(sample[2]))
#     make_folder(dest_color)
#
#     in_file = os.path.join(SLP_PATH, '{:05d}'.format(sample[0]), 'RGB', sample[1], 'image_{:06d}.png'.format(sample[2]))
#
#     copyfile(in_file, dest_filename)
#
#     # Write depth image
#     # PROX needs 8000 counts is 1 meter
#     depth, jtB, bbB = SLP_dataset.get_array_joints(idx_smpl=idx, mod='depthRaw')
#     depth = depth * 8
#
#     dest_depth = os.path.join(dest_folder, 'Depth')
#     make_folder(dest_depth)
#     dest_file_depth = os.path.join(dest_depth, 'image_{:06d}.png'.format(sample[2]))
#     cv2.imwrite(dest_file_depth, depth)


def copy_keypoints(sample, idx):
    # Openpose 25 keypoints described here https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
    # Convert the SLP 14-points to coco25 format

    depth, jtB, bbB = SLP_dataset.get_array_joints(idx_smpl=idx, mod='depthRaw')    # Keypoints are XY
    phy_vec = SLP_dataset.get_phy(idx=idx)

    # img = depth
    # for j in range(jtB.shape[0]):
    #     c = np.rint(jtB[j, :]).astype(np.int)
    #     img[c[1]:c[1] + 5, c[0]:c[0] + 5] = 0
    # plt.imshow(img)
    # plt.show()

    if WARP_TO_PM:
        h = henry_get_depth_homography(SLP_dataset, idx)
        new_joints = apply_homography(jtB[:, :2], h, False)
        jtB[:, :2] = new_joints

    slp_to_coco25 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 0]     # What I think it should be
    joints_coco25 = np.zeros((25, 3))
    for i in range(len(slp_to_coco25)):
        joints_coco25[slp_to_coco25[i], :2] = jtB[i, :2]
        joints_coco25[slp_to_coco25[i], 2] = 1  # Set confidence

    anno_dict = {'pose_keypoints_2d': joints_coco25.tolist()}
    anno_dict.update(phyvec_to_dict(phy_vec))
    full_dict = {'people': [anno_dict]}     # Wrap in the correct format

    dest_folder = os.path.join(OUT_PATH, 'keypoints', '{}_{:05d}'.format(sample[1], sample[0]))
    make_folder(dest_folder)
    dest_filename = os.path.join(dest_folder, 'image_{:06d}_keypoints.json'.format(sample[2]))
    # Target filename: /home/patrick/bed/prox/slp_tform/keypoints/00001_uncover/image_000001_keypoints.json

    with open(dest_filename, 'w') as outfile:
        json.dump(full_dict, outfile, indent=2)

    return joints_coco25


def make_depth(sample, idx, keypoints=None, vis=False):
    # Make a "mask" of the person by fitting a plane to the point cloud, removing outliers,
    # and getting a sharp depth image.

    depth, jt, bb = SLP_dataset.get_array_joints(idx_smpl=idx, mod='depthRaw', if_sq_bb=False)
    bb = bb.round().astype(int)
    bb += np.array([-25, -5, 50, 10])    # Patrick, expand "bounding box", since it often cuts off parts of the body

    if WARP_TO_PM:
        pointcloud = henry_convert_depth_2_pc(SLP_dataset, depth, idx)  # This isn't great since there's interpolation
        valid_pcd = np.logical_and(pointcloud[:, 2] > 1.55, pointcloud[:, 2] < 2.15)  # Cut out any outliers above the bed
        pointcloud = pointcloud[valid_pcd, :]

        # img = henry_get_warped_img(depth, henry_get_depth_homography(SLP_dataset, idx))
        # for j in range(keypoints.shape[0]):
        #     c = np.rint(keypoints[j, :]).astype(np.int)
        #     img[c[1]:c[1] + 5, c[0]:c[0] + 5] = 0
        # plt.imshow(img)
        # plt.show()
    else:
        pointcloud = ut.get_ptc(depth, SLP_dataset.f_d, SLP_dataset.c_d, bb) / 1000.0

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.020, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model

    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0.0, 0, 1.0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    plane_eqation = a * pointcloud[:, 0] + b * pointcloud[:, 1] + c * pointcloud[:, 2] + d
    pointcloud_sel = pointcloud[plane_eqation < BED_INLIER_THRESH, :]     # Only allow points a threshold above the plane
    pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud_sel))

    if vis:
        if WARP_TO_PM:
            camera = PerspectiveCamera(focal_length_x=367.8, focal_length_y=367.8, center=torch.Tensor(WARPED_DEPTH_TO_PM_CENTER).unsqueeze(0))
        else:
            camera = PerspectiveCamera(focal_length_x=367.8, focal_length_y=367.8, center=torch.Tensor([208.1, 259.7]).unsqueeze(0))
        all_joints = []
        joints_3d = camera.inverse_camera_tform(torch.Tensor(keypoints[:, :2]).unsqueeze(0), 2).detach().numpy()
        for i in range(25):
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.07)
            sph.compute_vertex_normals()
            sph.paint_uniform_color([0.0, 1.0, 0.0])
            sph.translate(joints_3d[0, i, :])
            all_joints.append(sph)

        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud] + all_joints)

    # cl, ind = pcd2.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)  # Remove outliers
    cl, ind = pcd2.remove_radius_outlier(nb_points=20, radius=0.05)  # Remove outliers
    inlier_cloud = pcd2.select_by_index(ind)
    inliers_numpy = np.asarray(inlier_cloud.points)

    if vis:
        o3d.visualization.draw_geometries([inlier_cloud])

    dest_folder = os.path.join(OUT_PATH, 'recordings', '{}_{:05d}'.format(sample[1], sample[0]))

    # Save raw pointcloud as numpy array
    ptc_filename = os.path.join(dest_folder, 'Depth', 'image_{:06d}_raw_ptc.npy'.format(sample[2]))
    np.save(ptc_filename, inliers_numpy)     # Save as meters


def make_dataset():
    all_samples = SLP_dataset.pthDesc_li

    for idx, sample in enumerate(tqdm(all_samples)):
        print(sample)
        # copy_images(sample, idx)
        keypoints = copy_keypoints(sample, idx)
        make_depth(sample, idx, keypoints=keypoints, vis=False)


if __name__ == "__main__":
    class PseudoOpts:
        SLP_fd = '/home/patrick/datasets/SLP/danaLab'  # give your dataset folder here
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover']  # give the cover class you want here
    SLP_dataset = SLP_RD(PseudoOpts, phase='all')  # all test result

    print(SLP_dataset.pthDesc_li)

    make_dataset()
