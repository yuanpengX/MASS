# MASS

*MASS* is a deep learning framework for multi-species RNA modification analysis by integrating RNA modification data from multiple species. This is an instruction of prediction RNA modification using *MASS*.

## Dependencies

These repository has been tested on Ubuntu 16.04. We stronly recommend you to have Anaconda3 installed, which contants most of required packages for running this model.

### Must installed packages

- tensorflow  1.8.0

- tensorlayer 1.10.1

- pandas 0.12.0

- [CD-HIT](http://weizhongli-lab.org/cd-hit/)

## Data Preparation

- Download complete data from the line[#], and unzip the data to the data folder.
- You can preprocess your own data with `./src/pocessing_pipeline.py`, and move the processed datqa to `./data`


## Get Started

### Test with pre-trained models
- The trained multi-speices model were stored in `./checkpoint`
- run the scipt `src/main.py` to generate prediction on test data, it will also report AUPRC and AUPR on the given data. Example code:
```bash
cd src
python ./src/main.py --name mass --data full_data --sn 8 --mode test --fold 0 --gpu 0
```

### Train new models

- You can train mass with you own data

```bash
cd src
python ./src/main.py --name mass --data full_data --sn 8 --mode train --gpu 0
```
## Utility files

*data2fa.py* is used to convert dataset into fasta format

*main.py* is the main function that has the following arguments:
 
 - --name: experiment name
 - --gpu: gpu id, currently MASS can only run on single GPU
 - --fold: fold id just for cross validation
 - --data: data source for trainning for 
 - --sn: number of species
 - --fold:  index for species in `./src/config.sample_names`

