"""
MiMSI Vector Generation Utility
Used to create vectors utilized by the MiMSI model during training and evaluation of MSI status in 
Next-Gen Sequencing Results. Reads a list of microsatellite regions provided via command line argument
and creates an instance vector for every region from a tumor/normal pair of bam files.

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
May 2020

zieglerj@mskcc.org
(c) 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details
"""

import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import os
import errno
import sys
import warnings
import pysam
import traceback
import pkg_resources
from glob import glob

from .bam2tensor import Bam2Tensor


# Global variables
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def process(line, bam2tensor):
    """
    Process is the most granular conversion function. It converts a single 
    microsatellite into a 100 x L x 3 vector, where 100 is the downsampled
    coverage configured in Bam2Tensor. If the microsatellite does not meet
    the required thresholds for length, repeat unit length, number of repeats
    or coverage (None, None) is returned. Otherwise a tuple containing 
    the vector and its location is returned to the wrapper function
    """
    # line = line.decode('utf8').strip()
    chrom, start, repeat_unit_length, repeat_unit_binary, repeat_times = line.split(
        "\t"
    )[0:5]

    if not chrom.isdigit():
        return (None, None)

    if int(repeat_unit_length) == 1 and int(repeat_times) < 10:
        return (None, None)

    if int(repeat_unit_length) < 5 and int(repeat_times) < 5:
        return (None, None)

    end = int(start) + int(repeat_unit_length) * int(repeat_times)
    total_len = end - int(start)
    if total_len < 5 or total_len > 40:
        return (None, None)
   
    return (bam2tensor.createTensor(str(chrom), int(start), int(end)), [str(chrom), int(start), int(end)])


def process_wrapper(bam_filename, norm_filename, m_list, chunkStart, chunkSize, covg):
    """
    This is a wrapper function that executes "process" above for a given
    chunk of the microsatellites list. It compiles a list of all the 
    microsatellites successfully converted to a numpy vector and returns 
    it along with the location information of the loci
    """
    with open(m_list, "r") as ms_list:
        # only look at microsatellites assigned to this process (chunks)
        ms_list.seek(chunkStart)
        lines = ms_list.read(chunkSize).splitlines()

        results = []
        locations = []

        if lines is None:
            # return empty
            return (results, locations)

        bamfile = pysam.AlignmentFile(bam_filename, "rb")
        normalbamfile = pysam.AlignmentFile(norm_filename, "rb")
        bam2tensor = Bam2Tensor(bamfile, normalbamfile, covg)

        for line in lines:
            result, loc = process(line, bam2tensor)
            if result is not None:
                # We got one!!!
                results.append(result)
                locations.append(loc)
        return (results, locations)


def chunkify(fname, size):
    """
    Generic helper method to break up a file into many smaller chunks,
    Here we use it to break up the microsatellites list so that we can
    generate many different microsatellite vectors in parallel.
    """
    fileEnd = os.path.getsize(fname)
    with open(fname, "rb") as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(size, 1)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break


def convert_bam(bamfile, norm_filename, m_list, covg, cores):
    """
    This is the top level function that converts an entire tumor/normal pair of bam files 
    into a vector collection that MiMSI will utilize in subsequent steps. It is setup to run
    in a parallel processing environment, with cores specified as a command line arg in main
    The steps it goes through are as follows: 
        1. chunk the list of microsatellites were interested in so that they can be executed 
            in parallel
        2. create a process_wrapper job to handle each chunk
        3. wait for all chunks to complete
        4. combine the results of each chunk
        5. close and return the combined results
    """
    all_instances = []
    all_locations = []

    pool = mp.Pool(int(cores))
    jobs = []
    file_size = os.path.getsize(m_list)
    chunk_size = int( file_size / (cores ) )
    try:
        # create jobs
        for chunkStart, chunkSize in chunkify(m_list, chunk_size):
            jobs.append(
                pool.apply_async(
                    process_wrapper,
                    (bamfile, norm_filename, m_list, chunkStart, chunkSize, covg),
                )
            )
        # wait for all jobs to finish
        for job in jobs:
            result = job.get()
            if result is not None:
                all_instances = all_instances + result[0]
                all_locations = all_locations + result[1]

        # clean up
        pool.close()
        pool.terminate()
    except Exception as e:
        print("There was an exception during parallel processing.")
        raise

    return (all_instances, all_locations)


def save_bag(tumor_sample, label, data, locations, save_loc, normal_sample=None):
    """ 
    Save the final collection of microsatellite instances to disk to be used
    in later stages of MiMSI. Saves each sample in the following manner:
        {sample}_{label}_data.npy
        {sample}_{label}_locations.npy
    This final filename format is important for later stages of the pipeline,
    as the data loader will parse the underscore deliminated filename to determine
    the sample id and the label. Sample id does need to be unique.
    """
    # if no instances that met the coverage threshold were found, return
    if len(data) == 0:
        print(
            "Sample %s did not have any microsatellites above the required threshold level. \n",
            tumor_sample,
        )
        return

    # convert to numpy to save to disk
    data = np.array(data)

    # save the instance to disk as it's generated, this is very important when
    # generating a large number of samples, otherwise everything will explode when you try
    # to keep storing all your samples in memory
    if normal_sample is not None:
        tumor_sample = "_".join([tumor_sample, normal_sample])
    file_name = save_loc + "/" + tumor_sample + "_" + str(label) + "_" + "data"
    loc_file_name = save_loc + "/" + tumor_sample + "_" + str(label) + "_" + "locations"
    np.save(file_name, data)
    np.save(loc_file_name, locations)


def create_sample_level_data(
    tumor_bam,
    normal_bam,
    m_list,
    save_loc,
    covg,
    cores,
    case_id,
    norm_case_id,
    label=None,
):
    """
    Wrapper function that process a single tumor and normal pairs
    and calls downstream functions that tranforms the bam files
    into vector data
    """

    def name_check(sample_name):
        """
        Helper function to enforce sample name format
        """
        if sample_name is None:
            return
        if "_" in sample_name:
            raise Exception(
                "'_' character not allowed in sample name {}.".format(sample_name)
            )

    map(name_check, [case_id, norm_case_id])
    if label is None:
        label = -1
    try:
        # convert
        result = convert_bam(tumor_bam, normal_bam, m_list, covg, cores)
        data = result[0]  # the converted vector
        locations = np.array(result[1])  # the locations utilized

        # save to disk
        save_bag(case_id, label, data, locations, save_loc, norm_case_id)

    except Exception:
        raise


def create_data(
    m_list, save_loc, covg, cores, case_list, tumor_bam, normal_bam, tumor_id, normal_id
):
    """
    Main wrapper function in this module that performs the
    pre-processing steps for single sample or batch mode 
    """
    print("Generating vectors for sample:")

    # create save directory, if one doesn't already exist
    try:
        os.mkdir(save_loc)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Exception when creating directory to store numpy array.")
            raise

    # remove existing data and locations npy arrays in the save_loc directory
    npy_files = glob(save_loc + "/*_locations.npy") + glob(save_loc + "/*_data.npy")
    if len(npy_files) > 0:
        try:
            for npy_file in npy_files:
                os.remove(npy_file)
        except OSError as e:
            if e.errno != errno.ENOENT:  # no such file or directory
                raise

    # If a cast list file is given use that to generate our data
    if case_list is not None and case_list != "":
        sample_list = []

        cases = pd.read_csv(case_list, sep="\t", header=0)
        case_list_cols = set(cases.columns.tolist())
        expected_case_list_cols = [
            "Tumor_ID",
            "Tumor_Bam",
            "Normal_Bam",
            "Normal_ID",
            "Label",
        ]
        if not case_list_cols <= set(expected_case_list_cols):
            raise Exception(
                "Column headers in the case list do not match expected values: {}. Note: Normal_ID column is optional.".format(
                    ",".join(expected_case_list_cols)
                )
            )

        # sanity checks
        if cases[["Tumor_ID", "Tumor_Bam", "Normal_Bam"]].isnull().values.any():
            raise Exception(
                "Missing/empty values in one or more of Tumor_ID, Tumor_Bam, Normal_Bam columns in the case list. Please check and try again."
            )

        if "Label" in case_list_cols and not set(
            cases[cases["Label"].notnull()].Label.apply(int).values.tolist()
        ) < set([+1, -1]):
            raise Exception(
                "Label column in case list can be empty or contain one the values +1 (MSI) or -1 (MSS)."
            )
        cases = cases.replace({np.nan: None})
        for index, row in cases.iterrows():
            tumor_id, normal_id, tumor_bam, normal_bam, label = map(
                lambda x: row[x] if x in row and row[x] else None,
                ["Tumor_ID", "Normal_ID", "Tumor_Bam", "Normal_Bam", "Label"],
            )
            if tumor_id in sample_list:
                raise Exception(
                    "Duplicate entires detected for sampleID {} in the case list".format(
                        tumor_id
                    )
                )
            sample_list.append(tumor_id)

            if label is None:
                label = -1
            else:
                label = int(label)

            print(str(index + 1) + ". " + tumor_id)
            create_sample_level_data(
                tumor_bam,
                normal_bam,
                m_list,
                save_loc,
                covg,
                cores,
                tumor_id,
                normal_id,
                label,
            )

    # Otherwise we are just going to convert the given sample
    # single sample mode will be only for evaluation. Therefore, label
    #  is not required. Will be defaulted to -1
    else:
        print("1. " + tumor_id)
        create_sample_level_data(
            tumor_bam, normal_bam, m_list, save_loc, covg, cores, tumor_id, normal_id
        )


def main():
    parser = argparse.ArgumentParser(description="MiMSI Vector Generation Utility")
    parser.add_argument(
        "--version",
        action="store_true",
        default=False,
        help="Display current version of MiMSI",
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
        default=None,
        help="Case List for generating sample vectors in bulk, if specified all other input file args will be ignored",
    )
    batch_mode_group.add_argument(
        "--name",
        default="BATCH",
        help="name of the run submitted using --case-list, this will be the filename for any saved results in the tsv format (default: BATCH)",
    )
    # Common arguments
    parser.add_argument(
        "--microsatellites-list",
        default=ROOT_DIR + "/../../tests/microsatellites_impact_only.list",
        help="The list of microsatellites to check in the tumor/normal pair (default: tests/microsatellites_impact_only.list)",
    )
    parser.add_argument(
        "--save-location",
        default="./generated_samples",
        help="The location on the filesystem to save the converted vectors (default: Current_working_directory/generated_samples/). WARNING: Existing files in this directory in the formats *_locations.npy and *_data.npy will be deleted!",
    )
    parser.add_argument(
        "--coverage",
        default=100,
        help="Required coverage for both the tumor and the normal. Any coverage in excess of this limit will be randomly downsampled",
    )
    parser.add_argument(
        "--cores", default=16, help="Number of cores to utilize in parallel"
    )

    args = parser.parse_args()

    if args.version:
        print("MiMSI Vector Creation CLI version - " + pkg_resources.require("MiMSI")[0].version)
        return 

    case_list, tumor_bam, normal_bam, case_id, m_list, save_loc, covg, cores, norm_case_id = (
        args.case_list,
        args.tumor_bam,
        args.normal_bam,
        args.case_id,
        args.microsatellites_list,
        args.save_location,
        args.coverage,
        args.cores,
        args.norm_case_id,
    )

    # sanity check
    if not any([case_list, tumor_bam]):
        raise Exception(
            "One of --case-list (batch mode) or --tumor-bam (single sample mode) must be defined."
        )

    create_data(
        m_list,
        save_loc,
        covg,
        cores,
        case_list,
        tumor_bam,
        normal_bam,
        case_id,
        norm_case_id,
    )


if __name__ == "__main__":
    main()

