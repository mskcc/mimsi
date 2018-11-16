# MiMSI

A deep, multiple instance learning based classifier for identifying Microsatellite Instability in Next-Generation Sequencing Results. 


Made with :heart: and :coffee: by ClinBx @ Memorial Sloan Kettering Cancer Center

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

The first column should contain a unique sample id, while the second and third columns specify the full filesystem path for the tumor and normal ```.bam``` files, respectively.

#### List of Microsatellite Regions

A list of microsatellite regions needs to be provided as a tab-seperated text file. A (very) short example list demonstrating the required columns is provided in the ```/utils/example_ms_list.txt``` file. The file we used in testing/training is available here (not linked yet).

### Running an individual sample

To run an individual sample,

```
python analyze.py --tumor-bam {/path/to/tumor.bam} --normal-bam {/path/to/normal.bam} --case-id my_unique_case --microsatellites-list {/path/to/microsatellites_file} --save-location {/path/to/save/vectors} --model ./model/mimsi_mskcc_impact.model > single_case_analysis.out
```

This pipeline can be run on both GPU and CPU setups. We've also provided an example lsf submission file - ```/utils/single-sample-full.lsf```. Just keep in mind that your institution's lsf setup may differ from our example.

### Running samples in bulk
Running a batch of samples is extremely similar, just provide a case list file rather than an individual tumor/normal pair,

```
python analyze.py --case-list {/path/to/case_list.txt} --microsatellites-list {/path/to/microsatellites_file} --save-location {/path/to/save/vectors} --model ./model/mimsi_mskcc_impact.model > multi_case_analysis.out
```

## Running Analysis Components Separately

### Vector Creation

### Evaluation

## Training the Model from Scratch

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details


