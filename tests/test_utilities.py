"""
MiMSI Test Script 

This testing script runs the full pipeline for the 100x and 200x model on an example tumor/normal
bam and verifies that the correct files are created at each point. It's run by the CI server in 
python 2.7, 3.5 and 3.6

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
Nov. 2018

zieglerj@mskcc.org

(c) 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details

"""


import numpy as np
import pandas as pd
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics
import traceback
import os
import gzip
import shutil

from data.generate_vectors.create_data import create_data
from main.evaluate_sample import run_eval

# Global variables
vector_folder = "./test_dir"
tumor_bam = "./test.tumor.bam"
normal_bam = "./test.normal.bam"
ms_list = "./microsatellites.list"
case_list = "./test_case_list.txt"


def setup():
    os.mkdir(vector_folder)


def teardown():
    for f in os.listdir(vector_folder):
        full_path = os.path.join(vector_folder, f)
        if os.path.isfile(full_path):
            os.unlink(full_path)

    os.rmdir(vector_folder)

    # if os.path.isfile("./test_results.npy"):
    #     os.unlink("./test_results.npy")


def test_100x_model():

    # test data creation
    create_data(
        ms_list, vector_folder, 50, 16, None, tumor_bam, normal_bam, "tumor", "normal"
    )

    assert os.path.isfile("./test_dir/tumor_normal_-1_data.npy")
    assert os.path.isfile("./test_dir/tumor_normal_-1_locations.npy")

    run_eval(
        "../model/mimsi_mskcc_impact.model",
        vector_folder,
        False,
        2,
        True,
        "both",
        vector_folder,
        "test-run",
        50,
        0.95,
    )

    assert os.path.isfile("./test_dir/tumor_normal_results.npy")
    assert os.path.isfile("./test_dir/test-run_results.txt")


def test_200x_model():

    # test data creation
    create_data(
        ms_list, vector_folder, 100, 16, None, tumor_bam, normal_bam, "tumor", "normal"
    )

    assert os.path.isfile("./test_dir/tumor_normal_-1_data.npy")
    assert os.path.isfile("./test_dir/tumor_normal_-1_locations.npy")

    run_eval(
        "../model/mimsi_mskcc_impact_200.model",
        vector_folder,
        False,
        2,
        True,
        "both",
        vector_folder,
        "test-run",
        100,
        0.95,
    )

    assert os.path.isfile("./test_dir/tumor_normal_results.npy")
    assert os.path.isfile("./test_dir/test-run_results.txt")


def batch_test():

    # test data creation
    create_data(ms_list, vector_folder, 50, 16, case_list, None, None, None, None)

    assert os.path.isfile("./test_dir/tumor1_normal1_-1_data.npy")
    assert os.path.isfile("./test_dir/tumor1_normal1_-1_locations.npy")
    assert os.path.isfile("./test_dir/tumor2_normal2_-1_data.npy")
    assert os.path.isfile("./test_dir/tumor2_normal2_-1_locations.npy")
    assert os.path.isfile("./test_dir/tumor3_-1_data.npy")
    assert os.path.isfile("./test_dir/tumor3_-1_locations.npy")

    run_eval(
        "../model/mimsi_mskcc_impact.model",
        vector_folder,
        False,
        2,
        True,
        "both",
        vector_folder,
        "test-run",
        50,
        0.95,
    )

    assert os.path.isfile("./test_dir/tumor1_normal1_results.npy")
    assert os.path.isfile("./test_dir/tumor2_normal2_results.npy")
    assert os.path.isfile("./test_dir/tumor3_results.npy")
    assert os.path.isfile("./test_dir/test-run_results.txt")
