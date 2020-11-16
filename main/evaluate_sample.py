"""
MiMSI Evalution Utility

Used to evaluate (test) a sample or samples for MSI status based on a tumor/normal vector

@author: John Ziegler
Memorial Sloan Kettering Cancer Center 
Nov. 2018

zieglerj@mskcc.org

(c) 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, 
and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, 
either version 3 or later. See the LICENSE file for details

"""
import os
import errno
import numpy as np
import scipy.stats as sps
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics

from data.data_loader import MSIBags
from model.mi_msi_model import MSIModel

# Global variables
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(model, eval_loader, cuda, save, save_format, save_loc, name, confidence):
    def sample_name_parser(sample):
        """
        Helper function to check whether both Tumor and
        Matched normal names can be retrieved from the
        sample name.
        """
        if sample.count("_") == 0:
            return [sample, ""]
        elif sample.count("_") > 1:
            raise Exception(
                (
                    "More than one '_' character in sample name {}. Expected one (if both tumor and normal IDs are included) or none (if only Tumor ID is included)."
                ).format(sample)
            )
        else:
            return sample.split("_")

    model.eval()
    sample_list = []
    results_nparray = np.empty([0, 10])

    with torch.no_grad():
        for batch_idx, (data, label, _, sample_id) in enumerate(eval_loader):
            # Since we're evaluating here we're just using a default label
            # of -1 and ignoring the loss
            bag_label = torch.tensor(int(label[0]))
            if cuda:
                bag_label = bag_label.cuda()
                Variable(bag_label)

            repeat_results = []
            for bag in data:
                if cuda:
                    bag = bag.cuda()
                    bag = Variable(bag)

                # Evaluate the sample
                _, Y_prob, Y_hat = model.calculate_objective(bag, bag_label)

                # Record the result as a probability
                Y_prob = Y_prob.item()
                repeat_results.append(Y_prob)

            # check for duplicate sample IDs
            if sample_id[0] in sample_list:
                raise Exception(
                    "Duplicate data detected for sampleID {}".format(sample_id[0])
                )
            sample_list.append(sample_id[0])
            results_nparray = np.append(results_nparray, [repeat_results], axis=0)

            print(sample_id[0] + "\t" + str(repeat_results))

    if save:
        try:
            os.mkdir(save_loc)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                print("Exception when creating directory to store final results.")
                raise

        if save_format in ["npy", "both"]:
            for sample, array in dict(zip(sample_list, results_nparray)).items():
                np.save(save_loc + "/" + sample + "_results.npy", array)

        if save_format in ["tsv", "both"]:
            if len(sample_list) == 1 and not name:
                name = sample_list[0]
            with open(save_loc + "/" + name + "_results.txt", "w") as f:
                f.write(
                    "\t".join(
                        [
                            "Tumor",
                            "Normal",
                            "Min_MSI_Score",
                            "Max_MSI_Score",
                            "Mean_MSI_Score",
                            "LCI",
                            "UCI",
                            "MSI_STATUS",
                        ]
                    )
                    + "\n"
                )
                for sample, array in dict(zip(sample_list, results_nparray)).items():
                    summary_stats = nparray_stats(array, confidence)
                    Tumor, Normal = sample_name_parser(sample)
                    f.write(Tumor + "\t" + Normal + "\t" + summary_stats + "\n")


def nparray_stats(nparray, confidence):
    """
    calculate and report summary stats for each numpy array/sample
    """

    def msi_status(msi_score):
        """
        helper function to assess MSI status
        """
        return "MSS" if msi_score <= 0.5 else "MSI"

    n = len(nparray)
    mean, se = np.mean(nparray), sps.sem(nparray)
    moe = se * sps.t.ppf((1 + confidence) / 2.0, n - 1)
    return "\t".join(
        map(
            lambda x: str(x),
            [
                np.min(nparray),
                np.max(nparray),
                mean,
                max(0.0, mean - moe),
                min(1.0, mean + moe),
                msi_status(mean),
            ],
        )
    )


def run_eval(
    saved_model,
    vector_location,
    cuda,
    seed,
    save,
    save_format,
    save_loc,
    name,
    coverage,
    confidence,
):
    """
    Main wrapper function to load the provided model and initiate evaluation
    """
    torch.manual_seed(seed)
    if cuda:
        print("\nGPU is Enabled!")

    print("Evaluating Samples, Lets go!!!")
    loader_kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
    eval_loader = data_utils.DataLoader(
        MSIBags(vector_location, int(coverage), False, False),
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )

    model = MSIModel(int(coverage))
    if cuda:
        model.cuda()

    model.load_state_dict(torch.load(saved_model))
    evaluate(model, eval_loader, cuda, save, save_format, save_loc, name, confidence)


def main():
    parser = argparse.ArgumentParser(description="MiMSI Sample(s) Evalution Utility")
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA for use off GPU, if this is not specified the utility will check availability of torch.cuda",
    )
    parser.add_argument(
        "--model",
        default=ROOT_DIR + "/../model/mimsi_mskcc_impact_200.model",
        help="name of the saved model weights to load",
    )
    parser.add_argument(
        "--vector-location",
        help="directory containing the generated vectors to evaluate",
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
        "--save-location",
        default="./mimsi_results",
        help="The location on the filesystem to save the final results (default: Current_working_directory/mimsi_results/).",
    )
    parser.add_argument(
        "--name",
        default="test_run_001",
        help="name of the run, this will be the filename for any saved results in tsv format with more than one samples.",
    )
    parser.add_argument(
        "--seed", type=int, default=2, metavar="S", help="Random Seed (default: 2)"
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

    args = parser.parse_args()

    saved_model, vector_location, no_cuda, seed, save, save_format, save_loc, name, coverage, confidence = (
        args.model,
        args.vector_location,
        args.no_cuda,
        args.seed,
        args.save,
        args.save_format,
        args.save_location,
        args.name,
        args.coverage,
        args.confidence_interval,
    )
    # Resolve args
    if save_loc == "./mimsi_results":
        try:
            save_loc = os.getcwd() + "/mimsi_results"
        except OSError as e:
            print("Cannot create directory to save final results!")
            print(traceback.format_exc())
            return False

    cuda = not no_cuda and torch.cuda.is_available()
    run_eval(
        saved_model,
        vector_location,
        cuda,
        seed,
        save,
        save_format,
        save_loc,
        name,
        coverage,
        confidence,
    )


if __name__ == "__main__":
    main()
