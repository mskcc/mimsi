'''
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

'''


import numpy as np

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

from data.generate_vectors.create_data import main as create_data
from main.evaluate_sample import main as run_eval

vector_folder = "./example"
tumor_bam = "./example.tumor.bam"
normal_bam = "./example.normal.bam"
ms_list = "./microsatellites.list"

def setup():
    os.mkdir(vector_folder)


def teardown():
    for f in os.listdir(vector_folder):
        full_path = os.path.join(vector_folder, f)
        if os.path.isfile(full_path):
            os.unlink(full_path)

    os.rmdir(vector_folder)

    if os.path.isfile("./example_results.npy"):
        os.unlink("./example_results.npy")


def test_100x_model():

    # test data creation
    create_data(None, tumor_bam, normal_bam, "example", ms_list, vector_folder, False, 50, 16)

    assert os.path.isfile("./example/example_-1_data.npy")
    assert os.path.isfile("./example/example_-1_locations.npy")

    run_eval("../model/mimsi_mskcc_impact.model", vector_folder, False, 2, True, "example", 50)

    assert os.path.isfile("./example_results.npy")


def test_200x_model():

    # test data creation
    create_data(None, tumor_bam, normal_bam, "example", ms_list, vector_folder, False, 100, 16)

    assert os.path.isfile("./example/example_-1_data.npy")
    assert os.path.isfile("./example/example_-1_locations.npy")

    run_eval("../model/mimsi_mskcc_impact_200.model", vector_folder, False, 2, True, "example", 100)

    assert os.path.isfile("./example_results.npy")