# Logbook to keep track of the progress of the project

# Mixture Density Networks
- Introduced by Bishop in 1994
- The main idea is to model the conditional probability distribution of the output given the input, that is, P(y|x).
- The usuall neural network models the conditional mean of the output given the input, that is, E[y|x]. Hence, they only provide a point estimate of the output. The problem with this is that the output may be multimodal, and the point estimate may not capture the distribution of the output.
- The idea is that a function can be represented as a mixture of kernels each with its own mean and variance and the mixture weights sum to 1.
- The mixture density network estimates all this parameters using a neural network. The output of the network is the parameters of the mixture distribution. Then we use the log-likelihood to train the network. One can show, as seen by Bishop, that in the
normal case of a NN, the log-likelihood gives the same cost function as the mean squared error. 

- The architecture of the network is the same as a normal NN, but the output layer is divided in three parts: the mean, the variance, and the mixture weights. The number of mixture components is a hyperparameter of the model.
- This 3 parts then use different activation functions due to the nature of the parameters:
  - The mean is a linear activation function since it can take any value.
  - The variance must be positive and we want a value far from zero because usually we divide by this value. Hence, some activation functions that can be used are a modified exponential function or the softplus function:
    - The ELU + 1 + error function is a good candidate since it is exponential for positive values and 0
    - The softplus function is a smooth approximation of the ReLU function and it is differentiable everywhere.
  - The mixture weights must be positive and sum to 1. Hence, we can use the softmax function.

- We can use Gaussians or other distributions as the kernels of the mixture. The most common is to use Gaussians. The mixture density network is then a generalization of the Gaussian Mixture Model.
