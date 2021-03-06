# MitoSplit-Net
The project main goal is to detect fission sites in mitochondria images. Previously, this was achieved by using a U-Net-like neural network with mitochondria (Mito) and Drp1 images as inputs. Drp1 is an enzyme responsible for the contrictions of mitochondiras, that finally lead to their division. To train the network, Dr. Dora Mahecic created a ground truth (GT) probability map of divisions based on a Hessian filter applied to the Mito channel. On top of that, she added the information about Drp1 concentration coming from the Drp1 channel plus some manual annotations. 

However, this model is computationally expensive and experimentally impractical, since it depends on imaging and processing both Mito and Drp1 channels. To address this problem, we developed MitoSplit-Net, a user-friendly library to create and evaluate new models that use only a Mito channel as input. This package is divided into the following modules:

## preprocessing.py
Preprocessing of the GT probability map of divisions. Includes tools for segmenting and analyzing fission sites. 

## augmentation.py
Functions adapted from the Python library albumentations.

## training.py
Create and train a model with the possibility of fine tuning its hyperparameters.

## evaluation.py
Study the performace of the network at both pixel and fission levels.

## util.py
Saving and loading processed images and models.

## plotting.py
Graphical tools to show input data overlapped with output data, compare two outputs, show performance metrics and more.

For more information and examples of each pipeline, please have a look at the notebook of the corresponding module.

## Contributions
We thank [@shedabnjm](https://github.com/shedabnjm) for the helpful discussion and input.
