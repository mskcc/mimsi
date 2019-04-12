# MiMSI

A deep, multiple instance learning based classifier for identifying Microsatellite Instability in Next-Generation Sequencing Results. 


Made with :heart: and lots of :coffee: by ClinBx @ Memorial Sloan Kettering Cancer Center

## Summary
Microsatellite Instability (MSI) is a phenotypic measure of deficiencies in DNA mismatch repair (MMR) machinery. These deficiencies lead to replication slippage in microsatellite regions, resulting in varying lengths of deletions in tumor samples. Detecting proper MSI status with high sensitivity and specificity in cancer patients is a critical priority in clinical genomics, especially after the FDA's recent approval of a targeted therapy for advanced cancer patients with MSI-high phenotype pan-cancer.

Current methods that determine MSI status using sequencing data compare the distributions of nucleotide deletion lengths between tumor and a matched normal with a statistical test. These methods lose sensitivity to accurately assess MSI status in some clinical situations, like low tumor purity samples or samples with low sequencing coverage. MiMSI is a multiple instance learning (MIL) model for predicting MSI phenotype from next-generation sequencing data that demonstrates very high sensitivity even in clinical situations where low purity samples are common.

## Getting Started

### Setup & Install

The source code and prebuilt model can be obtained by cloning this repo onto your local environment

```
git clone https://github.com/mskcc/mimsi.git
cd mimsi
export PYTHONPATH=$PYTHONPATH:/path/to/mimsi/:/path/to/mimsi/data:/path/to/mimsi/model
pip install -r requirements.txt
```

### Required Libraries

MiMSI is implemented in (Py)Torch using Python 2.7. We've included a requirements.txt file for use in pip. We recommend utilizing virtualenv, but feel free to use other environments like conda, local pip, etc.

Just note that the following packages are required:
* (Py)Torch
* Numpy
* Sklearn
* Pysam
  


To use the provided requirements.txt file, run:

```
pip install -r requirements.txt
```

## Running a Full Analysis

MiMSI is comprised of two main steps. The first is an NGS Vector creation stage, where the aligned reads are encoded into a form that can be interpreted by our model. The second stage is the actual evalution, where we input the collection of microsatellite instances for a sample into the pre-trained model to determine a classification for the collection. For convienence, we've packaged both of these stages into a python script that executes them together. If you'd like to run the steps individually (perhaps to train the model from scratch) please see the section "Running Analysis Components Separately" below.

More details on the MiMSI methods will be available in our pre-print, coming soon.


### Required files

MiMSI requires two main inputs - the tumor/normal pair of ```.bam``` files for the sample you would like to classify and a list of the microsatellite regions the model should check. 

#### Tumor & Normal .bam files

If you are only analyzing one file the tumor and normal bam files can be specified as command line args (see "Testing an Individual Sample" below). Additionally, if you would like to process multiple samples in one batch the tool can accept a tab-separated file listing each sample. Below is an example input file:

```
sample-id-1     /tumor/bam/file/location    /normal/bam/file/location
sample-id-2     /tumor/bam/file/location2    /normal/bam/file/location2
```

The first column should contain a unique sample id, while the second and third columns specify the full filesystem path for the tumor and normal ```.bam``` files, respectively. The vectors and final classification results will be named according to the sample id column, so be sure to use something that is unique and easily recognizable to avoid errors.

#### List of Microsatellite Regions

A list of microsatellite regions needs to be provided as a tab-seperated text file. A (very) short example list demonstrating the required columns is provided in the ```/utils/example_ms_list.txt``` file. A gzip compressed version of the file we used in testing/training is also available in the utils directory. These files were generated utilizing the excellent MSI Scan functionality from [MSISensor](https://github.com/ding-lab/msisensor).

### Running an individual sample

To run an individual sample,

```
cd /path/to/mimsi
python -m analyze --tumor-bam /path/to/tumor.bam --normal-bam /path/to/normal.bam --case-id ‘my-unique-case’ --microsatellites-list /path/to/microsatellites_file --save-location /path/to/save/vectors --model ./model/mimsi_mskcc_impact_200.model > single_case_analysis.out
```

The tumor-bam and normal-bam args specify the .bam files the pipeline will use when building the input vectors. These vectors will be saved to disk in the location indicated by the ```save-location``` arg. The format for the filename of the built vectors is ```{case-id}_data_{label}.npy``` and ```{case-id}_locations_{label}.npy```. The ```data``` file contains the N x coverage x 40 x 3 vector for the sample, where N is the number of microsatellite loci that were sucessfully converted to vectors. The ```locations``` file is a list of all the loci used to build the ```data``` vector, with the same ordering. These locations are saved in the event that you'd like to investigate how different loci are processed. The label is defaulted to -1 for evaluation cases, and won't be used in any reported calculations. The label assignment is only relevant for training/testing the model from scratch (see below section on Training). The results of the classifier are printed to standard output, so you can capture the raw probability score by saving the output as shown in our example (single_case_analysis.out). If you want to do further processing on results, add the ```--save``` param to automatically save the classification score to disk. The evaluation script repeats 10x for each sample, that way you can create a confidence interval for each prediction.

This pipeline can be run on both GPU and CPU setups. We've also provided an example lsf submission file - ```/utils/single-sample-full.lsf```. Just keep in mind that your institution's lsf setup may differ from our example.

### Running samples in bulk
Running a batch of samples is extremely similar, just provide a case list file rather than an individual tumor/normal pair,

```
cd /path/to/mimsi
python -m analyze --tumor-bam --case-list /path/to/case_list.txt --microsatellites-list /path/to/microsatellites_file --save-location /path/to/save/vectors --model ./model/mimsi_mskcc_impact_200.model > single_case_analysis.out
```
The NGS vectors will be saved in ```save-location``` just as in the single case example. Again, as in the single case example all results are printed to standard out and can be saved to disk by setting the optional ```--save``` flag.

## Running Analysis Components Separately

The analysis pipeline automates these two steps, but if you'd like to run them individually here's how.

### Vector Creation
In order to utilize our NGS alignments as inputs to a deep model they need to be converted into a vector. The full process will be outlined in our paper if you'd like to dive into the details, but for folks that are just interested in creating the data this step can be run individually via the ```data/generate_vectors/create_data.py``` script.

```
cd /path/to/mimsi
python -m data.generate_vectors.create_data --case-list /path/to/case_list.txt --cores 16  --microsatellites-list /path/to/microsatellites_list.txt --save-location /path/to/save/vectors --coverage 100 --is-labeled 1 > generate_vector.out
```
The ```--is-labeled``` flag is especially important if you plan on using the built vectors for training and testing. If you're only evaluating a sample a default label of -1 will be used, but if you want to train/test you'll need to provide that information to the script so that a relavant loss can be calculated. Labels can be provided as a fourth column in the case list text file, and should take on a value of -1 for MSS cases and 1 for MSI cases. Is also worth noting that this script was designed to be run in a parallel computing environment. The microsatellites list file is broken up into chunks, and separate threads handle processing each chunk. The microsatellite instance vectors for each chunk (if any) are then aggregated to create the full sample. You can modify the number of concurrent slots by setting the ```--cores``` arg. An example lsf submission script is available in ```utils/multi-sample-vector-creation.lsf```

### Evaluation
If you have a directory of built vectors you can run just the evalution step using the ```main/evaluate_samples.py``` script.

```
cd /path/to/mimsi
python -m main.evaluate_sample --saved-model ./model/mimsi_mskcc_impact_200.model --vector-location /path/to/generated/vectors --save --name "eval_sample" > output_log.txt
```

Just as with the ```analyze.py``` script output will be directed to standard out, and you can save the results to disk via the ```--save``` flag.

## Training the Model from Scratch
We've included a pretrained model file in the ```model/``` directory for evaluation, but if you'd like to try to build a new version from scratch with your own data we've also included our training and testing script in ```main/mi_msi_train_test.py```

```
cd /path/to/mimsi
python -m main.mi_msi_train_test --name 'model_name' --train-location /path/to/train/vectors/ --test-location /path/to/train/vectors/  --save 1 > train_test.log
```

Just note that you will need labels for any vectors utilized in training and testing. This can be accomplished by setting the ```--labeled``` flag in the Vector Creation script (see section above) and including the labels as a final column in the vector generation case list. The weights of the best performing model will be saved as a Torch ".model" file based on the name specified in the command args. Additionally, key metrics (TPR, FPR, Thresholds, AUROC) and train/test loss information are saved to disk as a numpy array. These metrics are also printed to standard out if you'd prefer to capture the results that way. Training a dataset of ~300 cases with ~1000 instances per case takes approximately 5 hours on 2 Nvidia DGX1 GPUs, see ```utils/train_test.lsf``` for an example lsf submission script.

## Questions, Comments, Collaborations?
Please feel free to reach out, I'm available via email - zieglerj@mskcc.org

If you have any issues or feature requests with the tool please don't hesitate to create an issue on this repo. We will address them as soon as possible.

Also, PRs are more than welcome. We'd love to see the community get involved and help utilize machine learning to improve cancer care!

## License

© 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 or later. See the [LICENSE](LICENSE) file for details


