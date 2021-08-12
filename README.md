# MitoSplit-Net
Detection of mitochondrial divisions with neural networks

The project main goal is to detect fission sites in mitochondria images. Previously, this was achieved by using a U-Net-like neural network with mitochondria (Mito) and Drp1 images as inputs. Drp1 is an enzyme responsible for the contrictions of mitochondiras, that finally lead to their division. To train the network, former PhD student Dora Mahecic created a ground truth for the probability map of divisions based on a Hessian filter applied to the Mito channel. On top of that, she added the information about Drp1 concentration coming from the Drp1 channel plus some manual annotations. 

However, this model is computationally expensive and experimentally impractical, since it depends on imaging and processing both Mito and Drp1 channels. In response to this problem, we developed MitoSplit-Net, a user-friendly library to create and evaluate new models that uses only the Mito channel as input. This package is divided into four parts:

- preprocessing
- augmentation
- training
- evaluation
