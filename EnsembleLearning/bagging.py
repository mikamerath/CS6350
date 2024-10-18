import sys

sys.path.insert(1, "../DecisionTree")

import ID3
import test_ID3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype


def main():
    depth = 16
    attributes = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y",
    ]
    train_df = pd.read_csv("bank-1/train.csv", names=attributes)
    test_df = pd.read_csv("bank-1/test.csv", names=attributes)
    test_df = prepare_test_df(test_df, attributes)
    prepared_train_df = prepare_test_df(train_df, attributes)
    attributes_map = {k: v for v, k in enumerate(attributes)}

    train_errors = []
    test_errors = []

    for T in range(1, 10):
        trees = bagging(train_df, T, attributes)
        train_errors.append(
            1 - predict(trees, prepared_train_df, ["yes", "no"], attributes_map)
        )
        test_errors.append(1 - predict(trees, test_df, ["yes", "no"], attributes_map))

    plot_errors(train_errors)
    plot_errors(test_errors)


def bagging(training_df, T, attributes, depth=16):
    ID3.metric = "info_gain"
    ID3.attributes_label = "y"
    ID3.max_depth = 16
    trees = []
    for i in range(T):
        samples = training_df.sample(len(training_df), replace=True)
        trees.append(ID3.ID3_prepare(samples, set(attributes[:-1]), "root", 0))

    return trees


def prepare_test_df(test_df, attributes):
    result = test_df.copy(deep=True)
    for attribute in attributes:
        if is_numeric_dtype(result[attribute]):
            median = result[attribute].median()
            result[attribute] = result[attribute].apply(
                lambda x: 1 if x >= median else 0
            )

    return result


def predict(trees, test_df, possible_outputs, a_map):
    correct = 0
    df_length = len(test_df)
    tree_length = len(trees)
    for row in test_df.values:
        prediction = 0
        for tree in trees:
            if test_ID3.test_observation(tree, row, possible_outputs, a_map):
                prediction += 1
        if prediction > tree_length / 2:
            correct += 1

    print(f"Error: {1 - correct/df_length}")
    return correct / df_length


def plot_errors(error_list):
    plt.title("Error over time")
    plt.xlabel("# Iterations")
    plt.ylabel("Error")
    plt.plot(range(0, len(error_list)), error_list)
    plt.show()


if __name__ == "__main__":
    main()
