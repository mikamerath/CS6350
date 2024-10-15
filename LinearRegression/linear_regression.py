import pandas as pd
import numpy as np


def main():
    df_cols = [
        "Cement",
        "Slag",
        "Fly ash",
        "Water",
        "SP",
        "Coarse Aggr",
        "Fine Aggr",
        "output",
    ]
    train_df = pd.read_csv("concrete/train.csv", names=df_cols)
    test_df = pd.read_csv("concrete/test.csv", names=df_cols)
    get_analytical_weight_vector(train_df)


def gradient_descent():
    print("Gradient")


def stochastic_gradient_descent():
    print("Hi")


def get_analytical_weight_vector(concrete_df):
    X = concrete_df.drop("output", axis=1)
    X["bias"] = 1
    X = X.to_numpy()
    X_transform = np.matrix.transpose(X)
    Y = concrete_df["output"].to_numpy()

    # Weights = (X^T * X)^-1 * X^T * Y
    result = np.matmul(
        np.matmul(np.linalg.inv(np.matmul(X_transform, X)), X_transform), Y
    )
    print(result)


if __name__ == "__main__":
    main()
