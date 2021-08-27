"""
MiMSI Analysis Utility

Used to run the full MiMSI pipeline, including vector generation. 

Vectors will be created from the provided bam file(s) and saved to disk. After completion all
vectors will be analyzed in bulk and the results reported 

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
Nov. 2018

zieglerj@mskcc.org

(c) 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details

"""

import os
import numpy as np
import scipy.stats as sps
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics
import traceback
import pkg_resources
from data.generate_vectors.create_data import create_data
from main.evaluate_sample import run_eval

# Global variables
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="MiMSI Analysis")
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="Display current version of MiMSI",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA for use off GPU, if this is not specified the utility will check availability of torch.cuda",
    )
    parser.add_argument(
        "--model",
        default=ROOT_DIR + "/model/mimsi_mskcc_impact_200.model",
        help="name of the saved model weights to load (default: model/mimsi_mskcc_impact_200.model)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="save the results of the evaluation to a numpy array or a tsv text file",
    )
    parser.add_argument(
        "--save-format",
        choices=["tsv", "npy", "both"],
        default="tsv",
        help="save the results of the evaluation to a numpy array or as summary in a tsv text file or both",
    )
    parser.add_argument(
        "--seed", type=int, default=2, metavar="S", help="Random Seed (default: 2)"
    )
    parser.add_argument(
        "--microsatellites-list",
        default=ROOT_DIR + "/utils/microsatellites.list",
        help="The list of microsatellites to check in the tumor/normal pair (default: utils/microsatellites.list)",
    )
    parser.add_argument(
        "--save-location",
        default="./mimsi_results",
        help="The location on the filesystem to save the converted vectors and final results (default: Current_working_directory/mimsi_results/). WARNING: Exisitng files in this directory in the formats *_locations.npy and *_data.npy will be deleted!",
    )
    parser.add_argument(
        "--cores",
        default=16,
        help="Number of cores to utilize in parallel (default: 16)",
    )
    parser.add_argument(
        "--coverage",
        default=100,
        help="Required coverage for both the tumor and the normal. Any coverage in excess of this limit will be randomly downsampled",
    )
    parser.add_argument(
        "--confidence-interval",
        default=0.95,
        help="Confidence interval for the estimated MSI Score reported in the tsv output file (default: 0.95)",
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        default=False,
        help="Use attention pooling rather than average pooling to aggregate sample embeddings (default: False)",
    )

    single_sample_group = parser.add_argument_group("Single Sample Mode")
    single_sample_group.add_argument(
        "--tumor-bam", help="Tumor bam file for conversion"
    )
    single_sample_group.add_argument(
        "--normal-bam", help="Matched normal bam file for conversion"
    )
    single_sample_group.add_argument(
        "--case-id",
        default="TestCase",
        help="Unique identifier for the single sample/case submitted. This will be the filename for any saved results (default: TestCase)",
    )
    single_sample_group.add_argument(
        "--norm-case-id", default=None, help="Normal case name (default: None)"
    )

    batch_mode_group = parser.add_argument_group("Batch Mode")
    batch_mode_group.add_argument(
        "--case-list",
        default="",
        help="Case List for generating sample vectors in bulk, if specified all other input file args will be ignored",
    )
    batch_mode_group.add_argument(
        "--name",
        default="BATCH",
        help="name of the run submitted using --case-list, this will be the filename for any saved results in the tsv format (default: BATCH)",
    )

    args = parser.parse_args()
    if args.version:
        print("MiMSI Case Analysis CLI version - " + pkg_resources.require("MiMSI")[0].version)
        return 

    case_list, tumor_bam, normal_bam, case_id, norm_case_id, ms_list, save_loc, cores, saved_model, no_cuda, seed, save, save_format, name, covg, confidence, use_attention = (
        args.case_list,
        args.tumor_bam,
        args.normal_bam,
        args.case_id,
        args.norm_case_id,
        args.microsatellites_list,
        args.save_location,
        args.cores,
        args.model,
        args.no_cuda,
        args.seed,
        args.save,
        args.save_format,
        args.name,
        args.coverage,
        args.confidence_interval,
        args.use_attention,
    )
    cuda = not no_cuda and torch.cuda.is_available()

    # Resolve args
    if case_list:
        print("Case list is provided. Running in batch mode!")
    elif all([tumor_bam, normal_bam]):
        print("Running in single sample mode!")
    else:
        print(
            "Either a case_list (batch mode) or a tumor_bam/normal_bam pairs (single sample mode) is required."
        )
        return

    if save_loc == "./mimsi_results":
        try:
            save_loc = os.getcwd() + "/mimsi_results"
        except OSError as e:
            print(
                "Cannot create directory to save intermediate files and final results!"
            )
            raise

    try:
        # is_labled is false since this is an evaluation pipeline, 50 is the coverage
        create_data(
            ms_list,
            save_loc,
            covg,
            cores,
            case_list,
            tumor_bam,
            normal_bam,
            case_id,
            norm_case_id,
        )
    except Exception:
        raise

    try:
        run_eval(
            saved_model,
            save_loc,
            cuda,
            seed,
            save,
            save_format,
            save_loc,
            name,
            covg,
            confidence,
            use_attention
        )
    except Exception:
        raise

    print("Analysis Complete!")


if __name__ == "__main__":
    main()
