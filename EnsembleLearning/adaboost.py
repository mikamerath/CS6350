import sys

sys.path.insert(1, "../DecisionTree")

import ID3

import pandas as pd
import numpy as np


def main():
    depth = 2
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
    test_df = pd.read_csv("bank-1/test.csv")

    ada_boost(train_df, T=1)


def ada_boost(training_df, T):
    X_df = training_df.drop("y")
    Y_df = training_df["y"]
    initial_weights = np.array([1 / len(training_df)] * len(training_df))
    for i in range(1, T):
        # Update Weights
        print("idk")

    # Construct strong hypothesis from weak hypotheses


def update_weights(weights, df):
    print("idk")


if __name__ == "__main__":
    main()
