# This file pulls mean betas and the initial poses for each participant and pose

import sys

# Set this to the SLP github repo
sys.path.append('/home/patrick/bed/SLP-Dataset-and-Code')
import numpy as np
import open3d as o3d
import os
from data.SLP_RD import SLP_RD
import json
import utils.utils as ut  # SLP utils
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
import json

SLP_PATH = '/home/patrick/datasets/SLP/danaLab'
FITS_PATH = '/home/patrick/bed/prox/slp_fits'
STAGE_TWO_PKL = 'stage_two.pkl'
STAGE_TWO_ANNO_FILE = 'stage_two_anno.json'


def read_anno():
    all_samples = SLP_dataset.pthDesc_li

    with open(STAGE_TWO_ANNO_FILE) as f:
        anno = json.load(f)

    anno_totalizer = {}

    for idx, sample in enumerate(all_samples):
        result = anno[str(sample)]
        anno_totalizer[result] = anno_totalizer.get(result, 0) + 1

        if result > 100:
            print('failure', result, sample)

    print(anno_totalizer)


if __name__ == "__main__":
    class PseudoOpts:
        SLP_fd = SLP_PATH
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover']  # give the cover class you want here


    SLP_dataset = SLP_RD(PseudoOpts, phase='all')  # all test result

    read_anno()
