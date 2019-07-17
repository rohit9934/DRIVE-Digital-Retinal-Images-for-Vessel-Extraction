# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by Rohit Sharma, Abdul Mugeesh and Kanishk Nama
This file is used to give all paths in our directories to python files for processing in neural network
"""
import argparse
import json

import os
from bunch import Bunch

from configs.utils.utils import mkdir_if_not_exist


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file) 

    config = Bunch(config_dict) 

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.val_groundtruth_path = os.path.join("/home/dgxuser102/data/team34/experiments/experiments", config.exp_name, "dataset/validate/groundtruth/")
    config.val_img_path=os.path.join("/home/dgxuser102/data/team34/experiments/experiments", config.exp_name, "dataset/validate/origin/")
    config.train_groundtruth_path = os.path.join("/home/dgxuser102/data/team34/experiments", config.exp_name, "dataset/train/groundtruth/")
    config.train_img_path = os.path.join("/home/dgxuser102/data/team34/experiments/experiments", config.exp_name, "dataset/train/origin/")
    config.hdf5_path = os.path.join("/home/dgxuser102/data/team34/experiments/experiments", config.exp_name, "hdf5/")
    config.checkpoint = os.path.join("/home/dgxuser102/data/team34/experiments/experiments", config.exp_name, "checkpoint/")
    config.test_img_path=os.path.join("/home/dgxuser102/data/team34/experiments", config.exp_name, "test/origin/")
    config.test_gt_path=os.path.join("/home/dgxuser102/data/team34/experiments", config.exp_name, "test/groundtruth/")
    config.test_result_path = os.path.join("/home/dgxuser102/data/team34/experiments", config.exp_name, "test/result/")
    mkdir_if_not_exist(config.train_img_path)
    mkdir_if_not_exist(config.train_groundtruth_path)
    mkdir_if_not_exist(config.val_img_path)
    mkdir_if_not_exist(config.val_groundtruth_path)
    mkdir_if_not_exist(config.hdf5_path)
    mkdir_if_not_exist(config.checkpoint)
    mkdir_if_not_exist(config.test_img_path)
    mkdir_if_not_exist(config.test_gt_path)
    mkdir_if_not_exist(config.test_result_path)

    return config







