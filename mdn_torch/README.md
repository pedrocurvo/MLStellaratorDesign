# Machine Learning for Stellarator Design 
## Implementation with PyTorch

## Table of Contents
- [Machine Learning for Stellarator Design](#machine-learning-for-stellarator-design)
  - [Implementation with PyTorch](#implementation-with-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Project Architecture Overview](#project-architecture-overview)
  - [How to run the code](#how-to-run-the-code)

## Project Overview
To design a stellarator one must find the appropriate input parameters that will generate the desired characteristics of the device. The goal of this work is to use machine learning, e.g. a neural network, to map the desired plasma properties to a set of input parameters that will generate the desired stellarator. 

To achieve this, a code that leverages the near-axis expansion simplification has been developed that is now routinely used to design new machines.

Using this code, a dataset of configurations has been generated.

Then since the code (function) is not invertible, the approach that was taken was to train a probabilistic model, a Mixture Density Network (MDN), to map the desired plasma properties to the input parameters.

After training the model and performing fine tuning, a new dataset of configurations was generated using the trained model. The following approach was taken to generate the new configurations:
- Train the model on the dataset.
- Select the good configurations from the dataset, following a certain criteria.
- Randomly sample different values from different rows to generate new samples.
- Run this samples through the model to get the input parameters.
- Generate new stellarators using the input parameters and the pyQSC code.
- Repeat ...

With this approach we increased the number of good stellarators in the dataset in each iteration.

## Project Architecture Overview

The project is structured as follows:

```
.
├── images
├── models
├── runs
├── StellaratorDataSet
├── train_pipeline
├── utils
├── __init__.py
├── data_analysis.ipynb
├── data_loss.py
├── iterations_correlation.ipynb
├── iterations_histograms.ipynb
├── qsc_predictor.py
├── README.md
└── train_mdn_fcov.py
```

```images```: Directory containing the images/graphs coming from the notebooks.

```models```: Directory containing the dictionaries of the trained models and means, the models are resgistred with the date and time of the training.

```runs```: Directory containing the TensorBoard logs of the training.

```StellaratorDataSet```: Directory containing the modules used to load the dataset.

```train_pipeline```: Directory containing the modules used to train the model.

```utils```: Directory containing the utility functions.

```data_analysis.ipynb```: Jupyter notebook containing the analysis of the datasets and iterations.

```data_loss.py```: Module containing the calculation of the loss for the generation of new stellarators using trained models.

```iterations_correlation.ipynb```: Jupyter notebook containing the correlation analysis of the iterations.

```iterations_histograms.ipynb```: Jupyter notebook containing the histograms of the iterations.

```qsc_predictor.py```: Module containing the function to use the model to predict 
new stellarators, provides a better Human-Interface.

```README.md```: This file.



## How to run the code

To train the model, first you need to have a data directory containing the dataset in numpy format. The dataset should be a numpy array with the shape (n_samples, n_features), where n_samples is the number of stellarators and n_features is the number of features used to describe the stellarators.

To train the model, you can run the following command:

```bash
python3 train_mdn_fcov.py --batch_size batch_size --num_epochs num_epochs --learning_rate learning_rate 
```

Where:
```
batch_size: The batch size used for training the model.
num_epochs: The number of epochs used for training the model.
learning_rate: The learning rate used for training the model.
```
