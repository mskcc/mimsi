# MiMSI

A deep, multiple instance learning based classifier for identifying Microsatellite Instability in Next-Generation Sequencing Results. 


Made with :heart: and lots of :coffee: by the Clinical Bioinformatics Team @ Memorial Sloan Kettering Cancer Center

Preprint: https://www.biorxiv.org/content/10.1101/2020.09.16.299925v1

## Summary
Microsatellite Instability (MSI) is a phenotypic measure of deficiencies in DNA mismatch repair (MMR) machinery. These deficiencies lead to replication slippage in microsatellite regions, resulting in varying lengths of deletions in tumor samples. Detecting proper MSI status with high sensitivity and specificity in cancer patients is a critical priority in clinical genomics, especially after the FDA's approval of a targeted therapy for advanced cancer patients with MSI-high phenotype regardless of cancer type.

Current methods that determine MSI status using sequencing data compare the distributions of nucleotide deletion lengths between tumor and a matched normal with a statistical test. These methods lose sensitivity to accurately assess MSI status in some clinical situations, like low tumor purity samples or samples with low sequencing coverage. MiMSI is a multiple instance learning (MIL) model for predicting MSI phenotype from next-generation sequencing data that demonstrates very high sensitivity even in clinical situations where low purity samples are common.

## Getting Started


### Installing with conda
```
conda install mimsi -c pytorch
evaluate_sample --version
```
*Note that the conda package requires python 3.6, 3.7, or 3.8*

### Installing from source with pip

The source code and prebuilt model can be obtained by cloning this repo onto your local environment:

```
git clone https://github.com/mskcc/mimsi.git
cd mimsi
```

and then to install via pip:
```
cd mimsi # Root directory of the repository that includes the setup.py script.
pip install .
evaluate_sample --version
```


## Running a Full Analysis

MiMSI is comprised of two main steps. The first is an NGS Vector creation stage, where the aligned reads are encoded into a form that can be interpreted by our model. The second stage is the actual evaluation, where we input the collection of microsatellite instances for a sample into the trained model to determine a classification for the sample. For convenience, we've packaged both of these stages into a python script that executes them together. If you'd like to run the steps individually (perhaps to train the model from scratch) please see the section "Running Analysis Components Separately" below.

If you're interested in learning more about the implementation of MiMSI please check out our [pre-print](https://www.biorxiv.org/content/10.1101/2020.09.16.299925v1)!


### Required files

MiMSI requires three main inputs:

1. The tumor/normal pair of ```.bam``` files for the sample you would like to classify
2. A list of the microsatellite regions the model should check
3. The trained model to use in evaluating the sample or samples

#### Tumor & Normal .bam files

If you are only analyzing one sample, the tumor and normal bam files can be specified as command line args (see "Testing an Individual Sample"). If you would like to process multiple samples in one batch the tool can accept a tab-separated file listing each sample with the following headers:

```
Tumor_ID	Tumor_Bam	Normal_Bam	Normal_ID	Label
sample-id-1     /tumor/bam/file/location    /normal/bam/file/location   normal1	-1
sample-id-2     /tumor/bam/file/location2    /normal/bam/file/location2 normal2	-1
```
The columns can be in any order as long as the column headers match the headers shown above. Note that the character '_' is not allowed in Tumor_ID and Normal_ID values. The Normal_ID column/values are optional. If Normal_ID is given for a tumor/normal pair, the npy array and final results will include the Normal_ID. Otherwise, only Tumor_ID will be reported. The Tumor_Bam and Normal_Bam columns specify the full filesystem path for the tumor and normal ```.bam``` files, respectively. The label field is not necessary if you are evaluating cases, but it's included in the event that you would like to train & test a custom model on your own dataset (see "Training the Model from Scratch"). The vectors and final classification results will be named according to the Tumor_ID, and optionally, Normal_ID columns, so be sure to use something that is unique and easily recognizable to avoid errors.

#### List of Microsatellite Regions

A list of microsatellite regions needs to be provided as a tab-separated text file. A (very) short example list demonstrating the required columns is provided in the ```/utils/example_ms_list.txt``` file. A  version of the file we used in testing/training is also available in the utils directory as 'microsatellites_impact_only.list'. NOTE: The sites present in this file focus on the regions targeted by MSK-IMPACT, the NGS assay utilized in our institute. You may use this or feel free to focus on sites particular to your own panel. These files were generated utilizing the excellent MSI Scan functionality from [MSISensor](https://github.com/ding-lab/msisensor).

#### Trained Models
MiMSI was developed and tested with varying coverage levels and different pooling mechanisms. We have provided four fully-trained models in the models directory - two different coverage levels (100x and 200x combined tumor & normal) with two different pooling mechanisms (average and attention). If you would like to use the attention models (indicated with an "_attn" suffix) include the ```--use-attention``` flag when calling analyze or evaluate_sample. The coverage levels can be set via the ```--coverage``` cli arg. The 200x model is the default, and for running the 100x models use ```--coverage 50```. 


### Running an individual sample

To run an individual sample,

```
analyze --tumor-bam /path/to/tumor.bam --normal-bam /path/to/normal.bam --case-id my-unique-tumor-id --norm-case-id my-unique-normal-id --microsatellites-list ./test/microsatellites_impact_only.list --save-location /path/to/save/ --model ./model/mi_msi_v0_4_0_200x_attn.model  --use-attention --save
```

The tumor-bam and normal-bam args specify the .bam files the pipeline will use when building the input vectors. These vectors will be saved to disk in the location indicated by the ```save-location``` arg. WARNING: Existing files in this directory in the formats *_locations.npy and *_data.npy will be deleted! The format for the filename of the built vectors is ```{case-id}_data_{label}.npy``` and ```{case-id}_locations_{label}.npy```. The ```data``` file contains the N x coverage x 40 x 3 vector for the sample, where N is the number of microsatellite loci that were successfully converted to vectors. The ```locations``` file is a list of all the loci used to build the ```data``` vector, with the same ordering. These locations are saved in the event that you'd like to investigate how different loci are processed. The label is defaulted to -1 for evaluation cases, and won't be used in any reported calculations. The label assignment is only relevant for training/testing the model from scratch (see below section on Training). The results of the classifier are printed to standard output, so you can capture the raw probability score by saving the output as shown in our example (single_case_analysis.out). If you want to do further processing on results, add the ```--save``` param to automatically save the classification score to disk. The evaluation script repeats 10x for each sample, that way you can create a confidence interval for each prediction.


### Running samples in bulk
Running a batch of samples is extremely similar, just provide a case list file rather than an individual tumor/normal pair,

```
analyze --case-list /path/to/case_list.txt --microsatellites-list ./test/microsatellites_impact_only.list  --save-location /path/to/save/ --model ./model/mi_msi_v0_4_0_200x_attn.model  --use-attention --save
```

The NGS vectors will be saved in ```save-location``` just as in the single case example. Again, as in the single case example all results are printed to standard out and can be saved to disk by setting the optional ```--save``` flag. Using the```--save-format``` you can choose to save the final results as a numpy array or a .txt file (including summary statistics) or both. The default mode is to save as a .txt file.
 


## Running Analysis Components Separately

The analysis pipeline automates these two steps, but if you'd like to run them individually here's how.

### Vector Creation
In order to utilize our NGS alignments as inputs to a deep model they need to be converted into a vector. The full process is outlined in the manuscript linked above, but for folks that are just interested in creating the data this step can be run individually via the ```data/generate_vectors/create_data.py``` script.


```
create_data --case-list /path/to/case_list.txt --cores 16  --microsatellites-list ./test/microsatellites_impact_only.list --save-location /path/to/save --coverage 100 > generate_vector.out
```


The ```--is-labeled``` flag is especially important if you plan on using the built vectors for training and testing. If you're only evaluating a sample a default label of -1 will be used, but if you want to train/test you'll need to provide that information to the script so that a relevant loss can be calculated. Labels can be provided as a fourth column in the case list text file, and should take on a value of -1 for MSS cases and 1 for MSI cases. This script was designed to be run in a parallel computing environment. The microsatellites list file is broken up into chunks, and separate threads handle processing each chunk. The microsatellite instance vectors for each chunk (if any) are then aggregated to create the full sample. You can modify the number of concurrent slots by setting the ```--cores``` arg. 

### Evaluation
If you have a directory of built vectors you can run just the evaluation step using the ```main/evaluate_samples.py``` script.

```
evaluate_sample --saved-model ./model/mi_msi_v0_4_0_200x_attn.model  --use-attention --vector-location /path/to/generated/vectors --save --name "eval_sample" > output_log.txt
```

Just as with the ```analyze.py``` script output will be directed to standard out, and you can save the results to disk via the ```--save``` flag.

## Training the Model from Scratch
We've included a pre-trained model file in the ```model/``` directory for evaluation, but if you'd like to try to build a new version from scratch with your own data we've also included our training and testing script in ```main/mi_msi_train_test.py```


```
mi_msi_train_test --name 'model_name' --train-location /path/to/train/vectors/ --test-location /path/to/train/vectors/  --save 1 > train_test.log
```

Just note that you will need labels for any vectors utilized in training and testing. This can be accomplished by setting the ```--labeled``` flag in the Vector Creation script (see section above) and including the labels as a final column in the vector generation case list. The weights of the best performing model will be saved as a Torch ".model" file based on the name specified in the command args. Additionally, key metrics (TPR, FPR, Thresholds, AUROC) and train/test loss information are saved to disk as a numpy array. These metrics are also printed to standard out if you'd prefer to capture the results that way. 

## Visualizing Sites
If you'd like to visualize a microsatellite instance (or instances) we've bundled a QC tool that will output a pdf containing an image of each channel separately. For example, to produce an image of the microsatellite instance at position 99192743 on chromosome 15 and save it as testviz.pdf:

```
visualize_instance   --vector /path/to/case/vector.npy --locations /path/to/case/locations.npy --site '15,99192743,99192758' --output 'testviz'
```

You can also visualize multiple sites passing in a sites file rather than one individual location

```
visualize_instance   --vector /path/to/case/vector.npy --locations /path/to/case/locations.npy --site-list test_sites.csv --output 'testviz'
```

The ```--site-list``` file must be a list of sites in the format 'chr,start,end' with each site separated by a new line.
```
9,139400835,139400849
2,48032740,48032753
10,43595836,43595850
22,41545024,41545038
2,47635523,47635536
4,187627659,187627675
```


## Questions, Comments, Collaborations?
Please feel free to reach out, I'm available via [email](mailto:jziegler820@gmail.com)!

If you have any issues or feature requests with the tool please don't hesitate to create an issue on this repo. We will address them as soon as possible.

Also, PRs are more than welcome. We'd love to see the community get involved and help utilize machine learning to improve cancer care!

## License

© 2018 Memorial Sloan Kettering Cancer Center.  This program is free software: you may use, redistribute, and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 or later. See the [LICENSE](LICENSE) file for details


