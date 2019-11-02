# MASS

MASS is a deep learning framework for cross-species RNA modification analysis.

## Requirements

These repository has been tested on Ubuntu 16.04.

We stronly recommend you to have Anaconda3 installed.

### Must installed packages

tensorflow  1.13.1

tensorlayer 1.10.1

## Utility files

*data2fa.py* is used to convert dataset into fastq format

*processing_pipeline.py* is used to generate training ready data.

*main.py* is the main function that has the following arguments:
 
 - --name: experiment name
 - --gpu: gpu id, currently MASS can only run on single GPU
 - --fold: fold id just for cross validation
 - --data: data source for trainning
 - --sn: number of species

*config.py* contains basic configuration for training

*make_motif.py* generate motifs based on a trained model

