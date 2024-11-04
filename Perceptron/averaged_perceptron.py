import pandas as pd
import numpy as np


def main():
    cols = ["variance", "skewness", "curtosis", "entropy", "label"]
    train_df = pd.read_csv("bank-note/train.csv", names=cols)
    train_df.loc[train_df["label"] == 0, "label"] = -1

    test_df = pd.read_csv("bank-note/test.csv", names=cols)
    test_df.loc[test_df["label"] == 0, "label"] = -1

    weights = averaged_perceptron(train_df, 1)
    print(f"Final Weights: {weights}")
    print(f"Error Rate: {get_error(test_df, weights)}")


def averaged_perceptron(train_df, learning_rate):
    X = train_df.drop(columns=["label"])
    X["bias"] = 1
    X = X.to_numpy()
    Y = train_df["label"].to_numpy()
    weights = np.array([0] * len(X[0]), dtype=np.float64)
    averages = np.array([0] * len(X[0]), dtype=np.float64)

    for _ in range(10):
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(Y)
        for i, row in enumerate(X):
            prediction = weights.dot(row)
            if prediction < 0:
                prediction = -1
            else:
                prediction = 1
            if prediction != Y[i]:
                weights += learning_rate * (row * Y[i])
        averages += weights

    return averages


def get_error(test_df, weights):
    X = test_df.drop(columns=["label"])
    X["bias"] = 1
    X = X.to_numpy()
    Y = test_df["label"].to_numpy()
    num_errors = 0
    for i, row in enumerate(X):
        prediction = weights.dot(row)
        if prediction < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != Y[i]:
            num_errors += 1

    return num_errors / len(X)


if "__main__" == __name__:
    main()
