# MiMSI

A deep, multiple instance learning based classifier for identifying Microsatellite Instability in Next-Generation Sequencing Results. 


Made with :heart: and lots of :coffee: by ClinBx @ Memorial Sloan Kettering Cancer Center

## Getting Started

### Setup & Install

The source code and prebuilt model can be obtained by cloning this repo onto your local environment

```
git clone https://github.com/mskcc/mimsi.git
cd mimsi
export PYTHONPATH=$PYTHONPATH:{deployment_location}/mimsi/:{deployment_location}/mimsi/data:{deployment_location}/mimsi/model
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

MiMSI is comprised of two main steps. The first an NGS Vector creation stage, where the aligned reads are encoded into a form that can be interpreted by our model. The second stage is the actual evalution, where we input the collection of microsatellite instances for a sample into the pre-trained model to determine a classification for the collection. For convienence, we've packaged both of these stages into a python script that executes them together. If you'd like to run the steps individually (perhaps to train the model from scratch) please see the section "Running Analysis Components Separately" below.

For more details on the MiMSI methods check out our paper, available in preprint here (not linked yet).


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

A list of microsatellite regions needs to be provided as a tab-seperated text file. A (very) short example list demonstrating the required columns is provided in the ```/utils/example_ms_list.txt``` file. The file we used in testing/training is available here (not linked yet).

### Running an individual sample

To run an individual sample,

```
python analyze.py --tumor-bam {/path/to/tumor.bam} --normal-bam {/path/to/normal.bam} --case-id my-unique-case --microsatellites-list {/path/to/microsatellites_file} --save-location {/path/to/save/vectors} --model ./model/mimsi_mskcc_impact.model > single_case_analysis.out
```

The tumor-bam and normal-bam args specify the .bam files the pipeline will use when building the input vectors. These vectors will be saved to disk in the location indicated by the ```save-location``` arg. The format for the filename of the built vectors is ```{case-id}_data_{label}.npy``` and ```{case-id}_locations_{label}.npy```. The ```data``` file contains the N x 100 x 40 x 3 vector for the sample, where N is the number of microsatellite loci that were sucessfully converted to vectors. The ```locations``` file is a list of all the loci used to build the ```data``` vector, with the same ordering. These locations are saved in the event that you'd like to perform some analysis on how different loci are processed by the model. The label is just defaulted to -1 for evaluation cases, and won't be used in any reported calculations. The label assignment is only relevant for training/testing the model from scratch (see below section on Training). The results of the classifier are printed to standard output, so you can capture the raw probability score by saving the output as shown in our example (single_case_analysis.out). If you want to do further processing on results, add the ```--save``` param to automatically save the classification score to disk.

This pipeline can be run on both GPU and CPU setups. We've also provided an example lsf submission file - ```/utils/single-sample-full.lsf```. Just keep in mind that your institution's lsf setup may differ from our example.

### Running samples in bulk
Running a batch of samples is extremely similar, just provide a case list file rather than an individual tumor/normal pair,

```
python analyze.py --case-list {/path/to/case_list.txt} --microsatellites-list {/path/to/microsatellites_file} --save-location {/path/to/save/vectors} --model ./model/mimsi_mskcc_impact.model > multi_case_analysis.out
```
The NGS vectors will be saved in ```save-location``` just as in the single case example. Again, as in the single case example all results are printed to standard out and can be saved to disk by setting the optional ```--save``` flag.

## Running Analysis Components Separately

The analysis pipeline automates these two steps, but if you'd like to run them individually here's how.

### Vector Creation
In order to utilize our NGS alignments as inputs to a deep model they need to be converted into a vector. The full process is outlined in our paper if you'd like to dive into the details, but for folks that are just interested in creating the data this step can be run individually via the ```data/generate_vectors/create_data.py``` script.

```
cd mimsi/data/generate_vectors/
python create_data.py --case-list {path/to/case_list.txt}--microsatellites-list {path/to/microsatellites_list.txt} --save-location {/path/to/save/vectors} --is-labeled 1 > vector_creation.out
```
The ```--is-labeled``` flag is especially important if you plan on using the built vectors for training and testing. If you're only evaluating a sample a default label of -1 will be used, but if you want to train/test you'll need to provide that information to the script so that a relavant loss can be calculated. Labels can be provided as a fourth column in the case list text file, and should take on a value of -1 for MSS cases and 1 for MSI cases. Is also worth noting that this script was designed to be run in a parallel computing environment. The microsatellites list file is broken up into chunks, and separate threads handle processing each chunk. The microsatellite instance vectors for each chunk (if any) are then aggregated to create the full sample. You can modify the number of concurrent slots by setting the ```--cores``` arg. An example lsf submission script is available in ```utils/multi-sample-vector-creation.lsf```

### Evaluation
If you have a directory of built vectors you can run just the evalution step using the ```main/evaluate_samples.py``` script.

```
python evaluate_sample.py --saved-model {deployment_location}/mimsi/model/mimsi_mskcc_impact.model  --vector-location {/path/to/vectors} > eval_results.out
```

Just as with the ```analyze.py``` script output will be directed to standard out, and you can save the results to disk via the ```--save``` flag.

## Training the Model from Scratch
We've included a pretrained model file in the ```model/``` directory for evaluation, but if you'd like to try to build a new version from scratch with your own data we've also included our training and testing script in ```main/mi_msi_train_test.py```

```
python mi_msi_train_test.py --name 'model_name' --train-location /path/to/train/vectors/ --test-location /path/to/train/vectors/  --save 1 > train_test.log
```

Just note that you will need labels for any vectors utilized in training and testing. This can be accomplished by setting the ```--labeled``` flag in the Vector Creation script (see section above) and including the labels as a final column in the vector generation case list. The weights of the best performing model will be saved as a Torch ".model" file based on the name specified in the command args. Additionally, key metrics (TPR, FPR, Thresholds, AUROC) and train/test loss information are saved to disk as a numpy array. These metrics are also printed to standard out if you'd prefer to capture the results that way. Training a dataset of ~300 cases with ~1000 instances per case takes approximately 5 hours on 2 Nvidia GeForce GTX 1080 GPUs, see ```utils/train_test.lsf``` for an example lsf submission script.

## Questions, Comments, Collaborations?
Please feel free to reach out! I'm available via email - zieglerj@mskcc.org

If you have any issues or feature requests with the tool please don't hesitate to create an issue on this repo. We'll do the best we can to address your concerns!

Also, feel free to submit a PR. We'd love to see the community get involved and help utilize machine learning to improve cancer care!

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details


