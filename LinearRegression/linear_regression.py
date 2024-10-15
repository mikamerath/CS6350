import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


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

    # get_analytical_weight_vector(train_df)
    error_history, final_weights = gradient_descent(train_df, 0.01, test_df)
    test_X = test_df.drop("output", axis=1)
    test_X["bias"] = 1
    test_X = test_X.to_numpy()
    test_Y = test_df["output"].to_numpy()
    print(error_history[-5:])
    plot_errors(error_history)
    print("Info for batch gradient descent")
    print(f"Final weights: {final_weights}")
    print(f"Learning rate: 0.01")
    print(f"Test data error: {calculate_cost(test_X, test_Y, final_weights)}")


def gradient_descent(concrete_df, learning_rate, test_df):
    Y = concrete_df["output"].to_numpy()
    X = concrete_df.drop("output", axis=1)
    X["bias"] = 1
    X = X.to_numpy()
    weights = np.array([0] * (len(concrete_df.columns)))
    epochs = 0
    error_history = []
    while epochs < 100000:
        gradient = calculate_gradient_vector(X, Y, weights)
        new_weights = weights - learning_rate * gradient
        error_history.append(calculate_cost(X, Y, new_weights))
        weight_change = np.linalg.norm((new_weights - weights), ord=1)
        if weight_change < 1e-6:
            break
        weights = new_weights
        epochs += 1

    return error_history, weights


def calculate_cost(X, Y, weights):
    total_cost = 0
    for row, y in zip(X, Y):
        total_cost += (y - row.dot(weights)) ** 2
    total_cost /= 2
    return total_cost


def calculate_gradient_vector(X, Y, weights):
    gradient = np.array([0] * len(weights), dtype=np.float32)
    for weight_position in range(len(weights)):
        for row, y in zip(X, Y):
            gradient[weight_position] += (y - weights.dot(np.array(row))) * row[
                weight_position
            ]

    gradient *= -1

    return gradient


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


def plot_errors(error_list):
    plt.title("Error over time")
    plt.xlabel("# Iterations")
    plt.ylabel("Error")
    plt.plot(range(0, len(error_list)), error_list)
    plt.show()


if __name__ == "__main__":
    main()
