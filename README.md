# Machine Learning for Stellarator Design

## Table of Contents
- [Machine Learning for Stellarator Design](#machine-learning-for-stellarator-design)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
  - [Project Architecture Overview](#project-architecture-overview)
  - [How to run the code](#how-to-run-the-code)

## Project Description
The design of fusion reactors with a stellarator configuration can be enormously simplified using a framework called the near-axis expansion. A Python code that leverages such simplification has been developed that is now routinely used to design new machines. From a set of input parameters, the code produces a new design, which can be assessed in terms of its characteristics, such as confinement and complexity of the geometry. However, to achieve a design with certain characteristics, one must find the appropriate input parameters. The goal of this work is to use machine learning, e.g. a neural network, to map the desired characteristics of the device to the corresponding parameters required to generate the device. This includes the development of a dataset of configurations and the training of the neural network on such dataset.

## Project Architecture Overview
The project is structured as follows:

```
.
├── mdn_keras
├── mdn_torch
├── resources
├── .gitignore
├── .gitattributes
├── main.py
└── README.md
```

```mdn_keras```: Directory containing the implementation of the project using Keras - TensorFlow.

```mdn_torch```: Directory containing the implementation of the project using PyTorch.

```resources```: Directory containing resources used in the project.

```main.py```: Main file to test the results (predict a Qsc object (stellarator) from desired outputs).

```README.md```: This file.


## How to run the code

1. Install required packages using the following command in the terminal:

    ```
    pip install -r requirements.txt
    ```

2. Go to the ´´´main.py´´´ and change the desired outputs to predict a Qsc object (stellarator) from them. Run the following command in the terminal:

    ```
    python3 main.py
    ```