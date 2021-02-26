# This file pulls mean betas and the initial poses for each participant and pose

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
import matplotlib.pyplot as plt
from prox.camera import PerspectiveCamera
from prox.misc_utils import smpl_to_openpose
import pprint
import argparse
from patrick_util import *


SLP_PATH = '/home/patrick/datasets/SLP/danaLab'
FITS_PATH = '/home/patrick/bed/prox/slp_fits'
STAGE_TWO_PKL = 'stage_two.pkl'


def get_smpl(pkl_data, json_data):
    gender = json_data['people'][0]['gender_gt']
    betas = torch.Tensor(pkl_data['betas']).unsqueeze(0)
    pose = torch.Tensor(pkl_data['body_pose']).unsqueeze(0)
    transl = torch.Tensor(pkl_data['transl']).unsqueeze(0)
    global_orient = torch.Tensor(pkl_data['global_orient']).unsqueeze(0)

    return {'gender': gender, 'betas': betas, 'pose': pose, 'transl': transl, 'global_orient': global_orient}


def view_fit(sample, idx):
    pkl_path = os.path.join(FITS_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'results', 'image_{:06d}'.format(sample[2]), '000.pkl')
    if not os.path.exists(pkl_path):
        print('MISSING', sample)
        return

    # print('Reading', pkl_path)
    pkl_np = pickle.load(open(pkl_path, 'rb'))

    json_path = os.path.join(FITS_PATH, 'keypoints', '{}_{:05d}'.format(sample[1], sample[0]), 'image_{:06d}_keypoints.json'.format(sample[2]))
    with open(json_path) as keypoint_file:
        json_data = json.load(keypoint_file)

    out_dict = get_smpl(pkl_np, json_data)
    return out_dict


def make_dataset():
    all_samples = SLP_dataset.pthDesc_li

    save_dict = {'all_results': {}, 'betas': {}}
    for idx, sample in enumerate(all_samples):
        out_dict = view_fit(sample, idx)
        key = tuple(sample)
        save_dict['all_results'][key] = out_dict

    for participant in range(1, 103):
        all_betas = np.zeros((45, 10))
        for sample in range(1, 46):
            all_betas[sample - 1, :] = save_dict['all_results'][(participant, 'uncover', sample)]['betas']

        save_dict['betas'][participant] = np.mean(all_betas, axis=0)

    pickle.dump(save_dict, open(STAGE_TWO_PKL, 'wb'))


if __name__ == "__main__":
    class PseudoOpts:
        SLP_fd = SLP_PATH
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover']  # give the cover class you want here
    SLP_dataset = SLP_RD(PseudoOpts, phase='all')  # all test result

    make_dataset()
