### Environment setup

**assuming you've already installed poetry and pyenv in your machine**

1. Download & utilize `.python-version`

    - `pyenv install $(pyenv latest -k X.Y)`
    - `pyenv local $(pyenv latest -k X.Y)`

2. Update `pyproject.toml` file 

    - `poetry env use $(cat .\.python-version)`

3. Install dependencies
    - `poetry install --no-root`

`cifar10_binary.ipynb` contains experiments using `svm.py` and `sklearn` for a binary classification problem (airplane vs rest)

`cifar10_multiclass.ipynb` containts experiments using `sklearn` for CIFAR-10 10-class classification problem on the full dataset

`mnist_multiiclass.ipynb` contains experiments using `sklearn` for MNIST 10-class classification problem on 1/3 of the full dataset



### `data_preprocessing_utils.py`

This module provides a set of utility functions to help load, preprocess, visualize, and handle image datasets such as CIFAR-10 and MNIST. The functions cover tasks including reading batch files, normalizing image data, displaying image grids, and constructing training/test sets.

### Features

- **Loading CIFAR-10 Batches:**  
  Load and decode CIFAR-10 batch files into Python dictionaries, converting binary keys to UTF-8 strings for easy manipulation.

- **Preprocessing Images:**  
  Reshape and normalize image data into a standardized format, ready for model training and evaluation.

- **Normalization Tools:**  
  Convert pixel values from `[0, 255]` to `[0, 1]` to improve training stability and performance of machine learning models.

- **Visualization:**  
  Display images in a grid for quick qualitative assessments, and show histograms of class distributions to understand dataset balance.

- **MNIST dataset Data Handling:**  
  Create training and test datasets from MNIST-like data structures for digit classification tasks.

### `model_training_utils.py`

This module provides functions for systematically exploring model hyperparameters through grid search, visualizing the results, and evaluating the best-found models. It leverages scikit-learn's cross-validation and parameter grid utilities, enabling an efficient and reproducible model selection process.

## Features

- **Grid Search for Hyperparameters:**  
  Iterates over all specified parameter combinations for a given model, using 5-fold stratified cross-validation to measure model performance. Records training and validation accuracies, as well as training times, to help you select the best parameters.

- **Result Visualization:**  
  Offers utilities to plot training and validation accuracies, as well as training times, against various model parameters. This provides intuitive insights into how model performance and computational cost scale with parameter choices.

- **Model Evaluation:**  
  Once the best parameters are found, the module can refit the model on the full training set and evaluate it on a test set, reporting final performance metrics and timing.


### `svm.py`

This module provides a custom Support Vector Machine (SVM) implementation called `MoschosSVM`. It supports a variety of kernel functions, including linear, polynomial, radial basis function (RBF), and multi-layer perceptron (MLP or sigmoid) kernels. The class leverages quadratic programming (via `cvxopt`) to solve the underlying optimization problem characteristic of SVMs.

By allowing users to set different kernels and tune parameters such as `C`, `gamma`, `degree`, and `coef0`, `MoschosSVM` offers a flexible platform for exploring different decision boundaries and fitting complexities. It provides low-level insight into the SVM process, making it useful for educational purposes or for scenarios where you want more control over the SVM training procedure than a high-level library might offer.


**This Class Includes:**
- A choice of kernels: `linear`, `poly` (polynomial), `rbf` (Radial Basis Function), and `mlp` (sigmoid-like).
- Parameter customization:
  - `C`: Regularization parameter controlling the trade-off between margin size and classification error.
  - `degree`: The degree of the polynomial kernel.
  - `gamma`: The kernel coefficient for `rbf`, `poly`, and `mlp` kernels.
  - `coef0`: An independent term added in polynomial and sigmoid kernels.
  
**Quadratic Programming Backend:**  
This implementation uses `cvxopt` to solve the dual SVM optimization problem. The solution yields Lagrange multipliers (alphas), support vectors, and the bias term `b`.
