import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import random


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

    test_X = test_df.drop("output", axis=1)
    test_X["bias"] = 1
    test_X = test_X.to_numpy()
    test_Y = test_df["output"].to_numpy()

    # Info about batch gradient descent
    error_history, final_weights = gradient_descent(train_df, 0.01)
    plot_errors(error_history)
    print("Info for batch gradient descent")
    print(f"Final weights: {final_weights}")
    print(f"Learning rate: 0.01")
    print(f"Test data error: {calculate_cost(test_X, test_Y, final_weights)}")

    # Info about stochastic gradient descent
    stochastic_error_history, stochastic_final_weights = stochastic_gradient_descent(
        train_df, 0.0002
    )
    plot_errors(stochastic_error_history)
    print("Info for batch gradient descent")
    print(f"Final weights: {stochastic_final_weights}")
    print(f"Learning rate: {0.0002}")
    print(
        f"Test data error: {calculate_cost(test_X, test_Y, stochastic_final_weights)}"
    )
    # Info about analytical solution
    print(
        f"Analytical linear regression result {get_analytical_weight_vector(train_df)}"
    )


def gradient_descent(concrete_df, learning_rate):
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


def stochastic_gradient_descent(concrete_df, learning_rate):
    Y = concrete_df["output"].to_numpy()
    X = concrete_df.drop("output", axis=1)
    X["bias"] = 1
    X = X.to_numpy()
    weights = np.array([0] * (len(concrete_df.columns)), dtype=np.float32)
    cost = float("inf")
    epochs = 0
    cost_history = []
    while epochs < 50000:
        weights = calculate_stochastic_gradient_vector(X, Y, weights, learning_rate)
        new_cost = calculate_cost(X, Y, weights)
        if abs(cost - new_cost) < 1e-9:
            break
        cost_history.append(new_cost)
        cost = new_cost

    return cost_history, weights


def calculate_gradient_vector(X, Y, weights):
    gradient = np.array([0] * len(weights), dtype=np.float32)
    for weight_position in range(len(weights)):
        for row, y in zip(X, Y):
            gradient[weight_position] += (y - weights.dot(np.array(row))) * row[
                weight_position
            ]

    gradient *= -1

    return gradient


def calculate_stochastic_gradient_vector(X, Y, weights, learning_rate):
    new_weights = np.copy(weights)
    random_row = random.randint(0, len(X) - 1)
    for weight_position in range(len(new_weights)):
        new_weights[weight_position] = float(
            (Y[random_row] - weights.dot(X[random_row]))
            * X[random_row][weight_position]
            * learning_rate
            + weights[weight_position]
        )

    return new_weights


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
    return result


def calculate_cost(X, Y, weights):
    total_cost = 0
    for row, y in zip(X, Y):
        total_cost += (y - row.dot(weights)) ** 2
    total_cost /= 2
    return total_cost


def plot_errors(error_list):
    plt.title("Error over time")
    plt.xlabel("# Iterations")
    plt.ylabel("Error")
    plt.plot(range(0, len(error_list)), error_list)
    plt.show()


if __name__ == "__main__":
    main()
