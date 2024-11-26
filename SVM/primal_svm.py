import numpy as np
import pandas as pd


def main():
    cols = ["variance", "skewness", "curtosis", "entropy", "label"]
    train_df = pd.read_csv("bank-note/train.csv", names=cols)
    train_df.loc[train_df["label"] == 0, "label"] = -1

    test_df = pd.read_csv("bank-note/test.csv", names=cols)
    test_df.loc[test_df["label"] == 0, "label"] = -1

    weights = primal_svm(train_df, learning_rate=1, C=100 / 873)
    print(weights)

    error = get_error(test_df, weights)
    print(error)


def primal_svm(train_df, learning_rate, C):
    X = train_df.drop(columns=["label"])
    X["bias"] = 1
    X = X.to_numpy()
    N = len(X)
    Y = train_df["label"].to_numpy()
    initial_weights = np.array([0] * len(X[0]), dtype=np.float64)
    weights = np.array([0] * len(X[0]), dtype=np.float64)

    for t in range(1, 100):
        # Use this for part 1
        gamma_t = learning_rate / (1 + learning_rate / 0.05 * t)

        # Use this for part 2
        gamma_t = learning_rate / (1 + t)
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(Y)
        for i, row in enumerate(X):
            prediction = Y[i] * weights.dot(row)
            if prediction <= 1:
                weights = (
                    weights - gamma_t * initial_weights + gamma_t * C * N * Y[i] * row
                )
            else:
                initial_weights = (1 - gamma_t) * initial_weights

            # print(weights)

    return weights


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


if __name__ == "__main__":
    main()
