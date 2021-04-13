import sys

# Set this to the SLP github repo
sys.path.append('/home/patrick/bed/SLP-Dataset-and-Code')
import numpy as np
import open3d as o3d
import os
from data.SLP_RD import SLP_RD
import json
import utils.utils as ut  # SLP utils
import pickle
import json
from pathlib import Path

SLP_PATH = '/home/patrick/datasets/SLP/danaLab'
FITS_PATH = '/home/patrick/bed/prox/slp_fits'
STAGE_TWO_PATH = '/home/patrick/bed/prox/slp_fits_two'
STAGE_THREE_PATH = '/home/patrick/bed/prox/slp_fits_three'
STAGE_TWO_ANNO_FILE = 'stage_two_anno.json'
STAGE_THREE_ANNO_FILE = 'stage_three_passed.json'
SAVE_PATH = '/home/patrick/bed/SLP_SMPL_fits'


def read_anno():
    all_samples = SLP_dataset.pthDesc_li

    with open(STAGE_TWO_ANNO_FILE) as f:
        stage_two_anno = json.load(f)

    with open(STAGE_THREE_ANNO_FILE) as f:
        stage_three_anno = json.load(f)

    for idx, sample in enumerate(all_samples):
        json_path = os.path.join(FITS_PATH, 'keypoints', '{}_{:05d}'.format(sample[1], sample[0]), 'image_{:06d}_keypoints.json'.format(sample[2]))
        with open(json_path) as keypoint_file:
            json_data = json.load(keypoint_file)

        out_dict = {}
        if str(sample) in stage_three_anno:
            pkl_path = os.path.join(STAGE_THREE_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'results', 'image_{:06d}'.format(sample[2]), '000.pkl')
            pkl_data = pickle.load(open(pkl_path, 'rb'))
            chosen_anno = stage_three_anno[str(sample)]
            pkl_chosen = pkl_data['all_results'][chosen_anno]
        elif str(sample) in stage_two_anno:
            pkl_path = os.path.join(STAGE_TWO_PATH, '{}_{:05d}'.format(sample[1], sample[0]), 'results', 'image_{:06d}'.format(sample[2]), '000.pkl')
            pkl_data = pickle.load(open(pkl_path, 'rb'))
            chosen_anno = stage_two_anno[str(sample)]

            if chosen_anno != 99:
                pkl_chosen = pkl_data['all_results'][chosen_anno]
            else:
                pkl_chosen = pkl_data
        else:
            print('Not found', sample)
            exit()

        out_dict['gender'] = json_data['people'][0]['gender_gt']
        out_dict['gt_joints'] = pkl_chosen['gt_joints']
        out_dict['betas'] = pkl_chosen['betas']
        out_dict['body_pose'] = pkl_chosen['body_pose']
        out_dict['transl'] = pkl_chosen['transl']
        out_dict['global_orient'] = pkl_chosen['global_orient']

        out_dict['camera_rotation'] = pkl_chosen['camera_rotation']
        out_dict['camera_translation'] = pkl_chosen['camera_translation']
        out_dict['camera_focal_length_x'] = pkl_chosen['camera_focal_length_x']
        out_dict['camera_focal_length_y'] = pkl_chosen['camera_focal_length_y']
        out_dict['camera_center'] = pkl_chosen['camera_center']

        save_pkl = os.path.join(SAVE_PATH, 'fits', 'p{:03d}'.format(sample[0]), 's{:02d}.pkl'.format(sample[2]))

        if not os.path.exists(os.path.dirname(save_pkl)):
            os.makedirs(os.path.dirname(save_pkl))

        pickle.dump(out_dict, open(save_pkl, 'wb'))


if __name__ == "__main__":
    class PseudoOpts:
        SLP_fd = SLP_PATH
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover']  # give the cover class you want here


    SLP_dataset = SLP_RD(PseudoOpts, phase='all')  # all test result

    read_anno()
