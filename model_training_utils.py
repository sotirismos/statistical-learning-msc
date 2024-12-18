from collections import defaultdict
from time import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, model_selection


def grid_search(model, param_dict, x_train, y_train):
    """
    Perform a grid search over given model parameters using cross-validation.

    This function iterates over all combinations of parameters provided
    in `param_dict` and evaluates the model's performance using stratified
    k-fold cross-validation. For each parameter set, it records the mean
    training and validation accuracies, as well as the mean training time.
    It returns a dictionary summarizing the results of the grid search,
    including the best parameters found.

    Parameters
    ----------
    model : estimator
        A scikit-learn compatible estimator. The estimator must implement
        `set_params` and `fit` methods. It can also implement `predict`,
        or for certain custom models like "MoschosSVM", a `pred` method.
    param_dict : dict
        A dictionary defining parameter ranges for the model. Keys are
        parameter names, and values are lists of parameter values to try.
    x_train : array-like of shape (n_samples, n_features)
        The training data input features.
    y_train : array-like of shape (n_samples,)
        The training data labels.

    Returns
    -------
    results : dict
        A dictionary containing:
        - "params": list of all parameter combinations tried.
        - "train_score": array of mean training accuracies for each parameter combination.
        - "val_score": array of mean validation accuracies for each parameter combination.
        - "time": array of mean training times for each parameter combination.
        - "best_index": int, the index of the best parameter combination.
        - "best_params": dict, the best parameter combination that achieved
            the highest validation accuracy.

    Notes
    -----
    This function uses 5-fold stratified cross-validation with a fixed
    random state for reproducibility.
    """
    param_grid = model_selection.ParameterGrid(param_dict)
    all_params = list(param_grid)
    num = len(all_params)
    train_acc = np.zeros(num)
    val_acc = np.zeros(num)
    t = np.zeros(num)

    cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, params in enumerate(param_grid):
        # Store scores for each fold
        val_acc_fold_scores = []
        train_acc_fold_scores = []
        fold_training_times = []

        # Store scores for each fold
        for train_index, val_index in cv.split(x_train, y_train):
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            model.set_params(**params)
            t1 = time()
            model.fit(x_train_fold, y_train_fold)
            t2 = time()
            fold_training_times.append(t2 - t1)

            if model.__class__.__name__ == "MoschosSVM":
                y_train_pred = model.pred(x_train_fold)
                y_val_pred = model.pred(x_val_fold)
            else:
                y_train_pred = model.predict(x_train_fold)
                y_val_pred = model.predict(x_val_fold)

            # Calculate accuracy
            train_acc_fold = metrics.accuracy_score(y_train_fold, y_train_pred)
            val_acc_fold = metrics.accuracy_score(y_val_fold, y_val_pred)

            val_acc_fold_scores.append(val_acc_fold)
            train_acc_fold_scores.append(train_acc_fold)

        vall_acc_folds_mean_score = np.mean(val_acc_fold_scores)
        train_acc_folds_mean_score = np.mean(train_acc_fold_scores)

        val_acc[i] += vall_acc_folds_mean_score
        train_acc[i] += train_acc_folds_mean_score
        t[i] = np.mean(fold_training_times)

        print(
            "[{}/{}] {}: train_acc = {:.4f}, val_acc = {:.4f} | t = {:.1f} sec = {:.1f} min".format(
                i + 1, num, params, train_acc[i], val_acc[i], t[i], t[i] / 60
            )
        )

    best_i = np.argmax(val_acc)
    best_params = all_params[best_i]

    print("Best params =", best_params)
    print("Maximum validation accuracy =", val_acc[best_i])

    results = {
        "params": all_params,
        "train_score": train_acc,
        "val_score": val_acc,
        "time": t,
        "best_index": best_i,
        "best_params": best_params,
    }

    return results


def plot_grid_search(results, param1, param2, xscale):
    """
    Plot the results of a grid search for two parameters.

    This function takes the results dictionary from the `grid_search` function
    and plots the training and validation accuracies as a function of one parameter,
    grouped by the values of another parameter. It also plots the training times
    for each parameter combination.

    Parameters
    ----------
    results : dict
        The dictionary returned by `grid_search`.
    param1 : str
        The primary parameter name to plot on the x-axis.
    param2 : str
        The secondary parameter name by which to group the plots. If this parameter
        is not present in `results["params"]`, the plots will be generated without grouping.
    xscale : str
        The scale of the x-axis (e.g., "linear", "log").

    Returns
    -------
    None
        Displays two plots: one showing training and validation accuracies,
        and another showing training times.
    """
    all_params = results["params"]
    train_acc = results["train_score"]
    val_acc = results["val_score"]
    t = results["time"]

    train_acc_dict = defaultdict(lambda: [])
    val_acc_dict = defaultdict(lambda: [])
    t_dict = defaultdict(lambda: [])
    values1_dict = defaultdict(lambda: [])

    for i, params in enumerate(all_params):
        if param2 in params:
            val2 = params[param2]
        else:
            val2 = None
        train_acc_dict[val2].append(train_acc[i])
        val_acc_dict[val2].append(val_acc[i])
        t_dict[val2].append(t[i])
        values1_dict[val2].append(params[param1])

    for val2, val_acc_vals in val_acc_dict.items():
        values1 = values1_dict[val2]
        train_acc_vals = train_acc_dict[val2]
        if val2 is None:
            train_label = "Training"
            val_label = "Validation"
        else:
            train_label = param2 + " = " + str(val2) + " (Training)"
            val_label = param2 + " = " + str(val2) + " (Validation)"
        plt.plot(values1, train_acc_vals, label=train_label, linestyle="dashed")
        plt.plot(values1, val_acc_vals, label=val_label)

    plt.title("Score")
    plt.ylabel("Accuracy")
    plt.xlabel(param1)
    plt.xscale(xscale)
    plt.legend()
    plt.show()

    for val2, t_vals in t_dict.items():
        values1 = values1_dict[val2]
        if val2 is not None:
            label = param2 + " = " + str(val2)
        else:
            label = None
        plt.plot(values1, t_vals, label=label)

    plt.title("Training time")
    plt.ylabel("Time (sec)")
    plt.xlabel(param1)
    plt.xscale(xscale)
    if param2 is not None:
        plt.legend()
    plt.show()


def evaluate_model(model_str, model, best_params, x_train, y_train, x_test, y_test):
    """
    Evaluate a model using the best parameters found by grid search.

    This function sets the model's parameters to the best found combination,
    trains the model on the entire training set, and evaluates its performance
    on both the training and test sets. It also records timing and returns
    a summary dictionary.

    Parameters
    ----------
    model_str : str
        A string identifier for the model (e.g., "SVC", "RandomForest").
    model : estimator
        The model or estimator to evaluate. Should implement `set_params`, `fit`,
        and `predict` or `pred` methods.
    best_params : dict
        The dictionary of parameter settings that resulted in the best
        performance during the grid search.
    x_train : array-like of shape (n_samples, n_features)
        The training data input features.
    y_train : array-like of shape (n_samples,)
        The training data labels.
    x_test : array-like of shape (n_samples, n_features)
        The test data input features.
    y_test : array-like of shape (n_samples,)
        The test data labels.

    Returns
    -------
    res : dict
        A dictionary containing:
        - "Classifier": The model identifier.
        - "Parameters": A string representation of the best parameters.
        - "Training Accuracy": The training accuracy as a formatted string.
        - "Test Accuracy": The test accuracy as a formatted string.
        - "Training Time (sec)": The training time in seconds as a formatted string.

    Notes
    -----
    This function also prints out the training and test accuracies, along
    with the training time. It does not display images or plots.
    """
    print("Training on the original training set with params =", best_params)
    model.set_params(**best_params)
    t1 = time()
    model.fit(x_train, y_train)
    t2 = time()
    print("Training time = {:.1f} sec = {:.1f} min".format(t2 - t1, (t2 - t1) / 60))
    if model.__class__.__name__ == "MoschosSVM":
        y_train_pred = model.pred(x_train)
        y_test_pred = model.pred(x_test)
    else:
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
    train_acc = metrics.accuracy_score(y_train, y_train_pred)
    test_acc = metrics.accuracy_score(y_test, y_test_pred)
    print("Training accuracy =", train_acc)
    print("Test accuracy =", test_acc)

    n = 5
    correct_indices = np.where(y_test == y_test_pred)[0]
    incorrect_indices = np.where(y_test != y_test_pred)[0]
    np.random.shuffle(correct_indices)
    np.random.shuffle(incorrect_indices)

    best_params_str = ""
    for param, value in best_params.items():
        if best_params_str != "":
            best_params_str += ", "
        best_params_str += param
        best_params_str += " = "
        if isinstance(value, float):
            best_params_str += "{:.4f}".format(value)
        else:
            best_params_str += str(value)

    res = {
        "Classifier": model_str,
        "Parameters": best_params_str,
        "Training Accuracy": "{:.4f}".format(train_acc),
        "Test Accuracy": "{:.4f}".format(test_acc),
        "Training Time (sec)": "{:.1f}".format(t2 - t1),
    }

    return res
