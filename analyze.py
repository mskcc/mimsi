'''
MiMSI Analysis Utility

Used to run the full MiMSI pipeline, including vector generation. 

Vectors will be created from the provided bam file(s) and saved to disk. After completion all
vectors will be analyzed in bulk and the results reported 

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
Nov. 2018

zieglerj@mskcc.org

© 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details

'''

from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics

from data.generate_vectors.create_data import main as create_data
from main.evaluate_sample import main as run_eval

def main(case_list, tumor_bam, normal_bam, case_id, ms_list, save_loc, cores, saved_model, no_cuda, seed, save, name):

    try:
        # is_lbled is false since this is an evaluation pipeline, 50 is the coverage
        create_data(case_list, tumor_bam, normal_bam, case_id, ms_list, save_loc, False, 50, cores)
    except Exception as e:
        print("There was an error generating the vectors: \n")
        print(e)
        return False

    try:
        run_eval(saved_model, save_loc, no_cuda, seed, save, name)
    except Exception as e:
        print("There was an error while evaluating samples: \n")
        print(e)
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MiMSI Analysis')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA for use off GPU, if this is not specified the utility will check availability of torch.cuda')
    parser.add_argument('--model', default="./model/mimsi_mskcc_impact.model", help='name of the saved model weights to load')
    parser.add_argument('--save', default=False, action='store_true', help='save the results of the evaluation to a numpy array')
    parser.add_argument('--name', default="test_run_001", help='name of the run, this will be the filename for any saved results')
    parser.add_argument('--seed', type=int, default=2, metavar='S', help='Random Seed (default: 2)')
    parser.add_argument('--case-list', default="", help='Case List for generating sample vectors in bulk, if specified all other input file args will be ignored')
    parser.add_argument('--tumor-bam', default="test-001-Tumor.bam", help='Tumor bam file for conversion')
    parser.add_argument('--normal-bam', default="test-001-Normal.bam", help='Matched normal bam file for conversion')
    parser.add_argument('--case-id', default="test-001", help='Unique identifier for the sample/case')
    parser.add_argument('--microsatellites-list', default="./microsatellites.list", help='The list of microsatellites to check in the tumor/normal pair')
    parser.add_argument('--save-location', default="./generated_samples", help='The location on the filesystem to save the converted vectors')
    parser.add_argument('--cores', default=16, help="Number of cores to utilize in parallel")
    args = parser.parse_args()


    main(args.case_list, args.tumor_bam, args.normal_bam, args.case_id, args.microsatellites_list, args.save_location, args.cores, args.model, args.no_cuda, args.seed, args.save, args.name)


