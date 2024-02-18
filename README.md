# Machine Learning for Stellarator Design

## Table of Contents
- [Machine Learning for Stellarator Design](#machine-learning-for-stellarator-design)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Project Architecture Overview](#project-architecture-overview)
  - [How to run the code](#how-to-run-the-code)
  - [Probabilistic model](#probabilistic-model)

## Project Description
The design of fusion reactors with a stellarator configuration can be enormously simplified using a framework called the near-axis expansion. A Python code that leverages such simplification has been developed that is now routinely used to design new machines. From a set of input parameters, the code produces a new design, which can be assessed in terms of its characteristics, such as confinement and complexity of the geometry. However, to achieve a design with certain characteristics, one must find the appropriate input parameters. The goal of this work is to use machine learning, e.g. a neural network, to map the desired characteristics of the device to the corresponding parameters required to generate the device. This includes the development of a dataset of configurations and the training of the neural network on such dataset.

## Project Architecture Overview
The project is structured as follows:

```
.
├── generate.py
├── CSVtoNumpyConverter.py
├── data
│   ├── dataset.csv
│   └── dataset.npy
├── StellatorsDataSet
│   ├── __init__.py
│   ├── StellaratorsDataSetDirect.py
│   └── StellaratorsDataSetInverse.py
├── train_pipeline
│   ├── data_setup.py
│   ├── engine.py
│   ├── MBuilder
│   │   ├── __init__.py
│   │   ├── MixtureDensityNetwork.py
│   │   └── ForwardNeuralNetwork.py
│   ├── utils.py
│   └── predictions.py
├── models
├── runs
├── train.py
├── resources 
├── requirements.txt
├── .gitignore
├── LICENSE
├── IDEAS.md
└── README.md
```

```data```: Directory containing the data set used for training and testing.

*  ```dataset.csv```: CSV file containing the dataset.
*  ```dataset.npy```: Numpy file containing the dataset.
    
```models```: Directory containing the dictionaries of the trained models, the models are resgistred with the date and time of the training.

```runs```: Directory containing the TensorBoard logs of the training.

```train_pipeline```: Directory containing the modules used to train the model.

 *   ```data_setup.py```: Module used to load the dataset and split it into training and testing sets, returning the data loaders.
 *   ```engine.py```: Module containing the training and testing loop functions.
 *   ```model_builder.py```: Module containg several model classes and model architecture.
 *   ```utils.py```: Module containing utility functions such as the function used to save the models,
 *   ```predictions.py```: Module used to generate predictions and evaluate models.

```train.py```: Entry point of the training pipeline, it is used to train the model and save it. Here the hyperparameters of the model can be set, such as the number of epochs, the learning rate, the batch size, loss fuction, optimizer, etc.

```StellatorsDataSet.py```: Class used to load the dataset.

```CSVtoNumpyConverter.py```: Script used to convert the dataset from CSV to Numpy format to be used by StellatorsDataSet class.

```generate.py```: Generator of stellarators, using the pyQSC package. It is used to generate the dataset.

```resources```: Directory containing resources such as papers, presentations, etc.. to give a theoretical background to the project.

```requirements.txt```: List of packages required to run the code.

```.gitignore```: List of files that should not be uploaded to the repository.

```LICENSE```: License of the project.

```README.md```: Description of the project.



## How to run the code

1. Install required packages using the following command in the terminal:

    ```
    pip install -r requirements.txt
    ```

2. Adjust hyperparameters in ```train.py``` and choose a model from ```model_builder.py```.

3. Run the code using the following command in the terminal:

    ```
    python3 train.py
    ```

4. Your model will be saved in the ```models/model.__class__.__name__``` directory with the date and time of the training (e.g. ```models/Model/2024_01_22_23_49_10.pth```).

5. A TensorBoard log will be created in the ```runs``` directory(e.g. ```runs/MLStellaratorDesign/Model/2024_01_22_23_49_10```).
6. To visualize the tensorboard log you can run the following command in the terminal:

    ```
    tensorboard --logdir=runs
    ```

7. A link will be provided in the terminal, which you can copy and paste in your browser to visualize the tensorboard log. 
```TensorBoard 2.15.1 at http://localhost:6007/ (Press CTRL+C to quit)```. In the tensorboard log you can visualize the train and test losses, accuracy if it is a classification problem, and other metrics... You can also visualize the model architecture and so on...

8. Alternatively, if you are using VSCode, you can install the TensorBoard extension and visualize the tensorboard log directly in VSCode by
pressing ```Cmd+Shift+P``` and typing ```Python: Launch TensorBoard ``` and selecting the dir or log you want to visualize.

## Probabilistic model

The probabilistic model is a multivariate normal distribution parameterized by a neural network (see `model.py`).

1. Generate the training data by running `generate.py`. This will generate a dataset in `data/dataset.csv`.

2. Train the model by running `model_train.py`. The model weights will be saved in `model_weights.h5`.

3. Predict "good" stellarators by running `model_predict.py`. The results will be saved in `data/predict.csv`.

The script `qsc_sampling.py` contains supporting routines for `generate.py` and `model_predict.py`.