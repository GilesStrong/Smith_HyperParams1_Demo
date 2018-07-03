[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/GilesStrong/Smith_HyperParams1_Demo/master)

# Hyper-Parameter Optimisation Part I
Demonstration of the techniques developed by L. Smith in [arXiv:1803:09820](https://arxiv.org/abs/1803.09820) - 
A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay. 
Originally presented at the journal club during the LIP-Lisbon Big-Data Meeting on 06/07/18

The four notebooks apply the various recommended techniques to a classifer acting on the data from the 2014 [Kaggle HiggsML challenge](https://www.kaggle.com/c/higgs-boson)

Data should be present if running via Binder, otherwise it can be downloaded from [here](http://opendata.cern.ch/record/328)

## References:
- Leslie N. Smith, A disciplined approach to neural network hyper-parameters: Part 1 - learning rate, batch size, momentum, and weight decay, CoRR, April, 2018, (http://arxiv.org/abs/1803.09820)
- ATLAS collaboration (2014). Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014. CERN Open Data Portal. [DOI:10.7483/OPENDATA.ATLAS.ZBP2.M5T8](http://opendata.cern.ch/record/328)

## Acknowledgements
- Some of the callbacks have been adapted from the Pytorch versions implemented in the [Fast.AI library](https://github.com/fastai/fastai), for use with Keras
