# Machine Learning for Stellarator Design

## Table of Contents
- [Machine Learning for Stellarator Design](#machine-learning-for-stellarator-design)
  - [Table of Contents](#table-of-contents)
  - [Project Description](#project-description)
    - [Stellarators and Near-Axis Expansion](#stellarators-and-near-axis-expansion)
    - [Mixture Density Networks](#mixture-density-networks)
    - [Predicting Good Stellarators](#predicting-good-stellarators)
  - [Project Architecture Overview](#project-architecture-overview)
  - [How to run the code](#how-to-run-the-code)
    - [main.py](#mainpy)
    - [generator.py](#generatorpy)

## Project Description
The design of fusion reactors with a stellarator configuration can be enormously simplified using a framework called the near-axis expansion. A Python code that leverages such simplification has been developed that is now routinely used to design new machines. From a set of input parameters, the code produces a new design, which can be assessed in terms of its characteristics, such as confinement and complexity of the geometry. However, to achieve a design with certain characteristics, one must find the appropriate input parameters. The goal of this work is to use machine learning, e.g. a neural network, to map the desired characteristics of the device to the corresponding parameters required to generate the device. This includes the development of a dataset of configurations and the training of the neural network on such dataset.

### Stellarators and Near-Axis Expansion
Stellarators are a type of magnetic confinement fusion device and are among the leading candidates for future power plants. Like tokamaks, they have toroidal geometry and flux-concentric surfaces.

However, their complex geometries can make it difficult to confine charged particles, especially alpha particles from fusion reactions. Therefore, they require accurately shaped magnetic fields to effectively confine these particles.

Achieving this requires optimizing their configurations using numerical methods. This optimization process is complex due to the high-dimensional space of plasma shapes, which includes many local minima. Local optimization algorithms can find specific configurations but do not provide a global view of the solution space. The high dimensionality also makes global optimization challenging and comprehensive parameter scans impractical.

To address these challenges, a near-axis method is commonly employed. This method uses an approximate magnetohydrodynamic (MHD) equilibrium model by expanding in powers of the distance to the axis. This approach significantly reduces computational cost and facilitates the generation of extensive databases of stellarator configurations. Examples of geometric parameters include rc1, rc2, zs1, zs2, zs3, etabar, and p2. Examples of confinement properties include axis length, iota, Dmerc_times_r2, beta, min L grad b, and L grad grad b.

### Mixture Density Networks
Nowadays, software packages that implement the near-axis method already exist, such as pyQSC, which takes a set of design parameters and computes various properties. However, not all configurations are desirable. Many sets of weights and targets can result in unacceptable configurations due to factors like insufficient plasma volume or excessive elongation. Therefore, it is crucial to verify whether the configurations meet specific criteria. This verification can be time-consuming and often requires running the near-axis method multiple times to achieve a viable configuration or resorting to numerical optimization. This raises the question of whether it is possible to perform inverse design, that is, to determine the input parameters from a given set of desired properties.

Due to the nature of the near-axis method equations, analytically inverting the problem is challenging. Therefore, using a neural network as a universal approximator is a better choice. However, this inverse problem is ill-posed since multiple inputs can produce the same set of output properties. This makes it a non-bijective function, and a standard neural network is not suitable, as it will predict the mean of the multiple inputs, which is not accurate.

To address this, we use a probabilistic model that can predict the distribution of the variables based on the desired properties. The model uses a mixture of several Gaussians to approximate the distribution, known as a mixture model. The parameters of these Gaussians—means, covariance matrices, and weights—are determined using a neural network. For each desired property, the neural network produces a set of parameters to create a distribution of the input properties. We then sample from this distribution to obtain the input parameters for the near-axis method. In this approach, the neural network maps the desired properties to the parameters of the distribution, functioning as a continuous function.

### Predicting Good Stellarators
Initially, we created a dataset featuring near-axis configurations, generated using uniform random distribution. Subsequently, we iteratively trained the model. In each iteration, we trained the model and then used it to predict new stellarators. In the beginning, we had only a small fraction of good stellarators. However, as the model learned, we observed a significant improvement in the prediction of good stellarators. Initially, through the random generation, we had roughly 1 in a million good stellarators, but by the final iteration of the model, this improved to 1 in 5.

This improvement demonstrates that the model is progressively learning the parameter distribution of good stellarators, making it more adept at identifying those with desirable characteristics. As a side result, we amassed a substantial dataset of good stellarators, enabling a thorough analysis of their parameter space. Notably, we discovered that a viable stellarator, suitable for construction, for example, typically has around three or four field periods. Beyond that, here is a numerical example. For a specific set of desired properties, the model suggests the following input parameters. When these parameters are applied using the near-axis method, the resulting output characteristics closely match the desired properties. This demonstrates the model’s ability to produce a stellarator that aligns well with the specified requirements.

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
├── generator.py
└── README.md
```

```mdn_keras```: Directory containing the implementation of the project using Keras - TensorFlow.

```mdn_torch```: Directory containing the implementation of the project using PyTorch.

```resources```: Directory containing resources used in the project.

```main.py```: Main file to test the results (predict a Qsc object (stellarator) from desired outputs).

```generator.py```: File that allows to generate stellarators using the
torch model trained in this project. 

```README.md```: This file.

More details could be found on the ```README.md``` files inside the directories ```mdn_keras``` and ```mdn_torch```.


## How to run the code

### main.py

1. Install required packages using the following command in the terminal:

    ```
    pip install -r requirements.txt
    ```

2. Go to the ´´´main.py´´´ and change the desired outputs to predict a Qsc object (stellarator) from them. Run the following command in the terminal:

    ```
    python3 main.py
    ```

### generator.py
1. You can generate stellarators using the torch model trained in this project. Run the following command in the terminal:

    ```
    python3 generator.py \
    --model "path/to/your/model.pth" \
    --model_mean "path/to/your/mean_std.pth" \
    --from_data "path/to/your/data.csv" \
    --to_data "path/to/save/new_data.csv" \
    --num_samples 50000
    ```
- All the arguments have default values, just be sure to download the data for the ```from_data``` argument. You can find the details for the arguments with the following command:
      
      python3 generator.py --help
      
