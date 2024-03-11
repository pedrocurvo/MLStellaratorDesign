## Probabilistic model

The probabilistic model is a mixture density network parameterized by a neural network (see `model.py`).

1. Generate the training data by running `generate.py`. This will generate a dataset in `dataset.csv`.

2. Train the model by running `model_train.py`. The model weights will be saved in `model_weights.h5`.

3. Predict "good" stellarators by running `model_predict.py`. The results will be saved in `predict.csv`.

The script `sampling.py` contains supporting routines for `generate.py` and `model_predict.py`.