import pandas as pd

from pandas.api.types import is_numeric_dtype

import ID3

# data_path = "car/"
all_attributes = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "label",
]
attributes_label = "label"

bank_data_path = "bank/"
bank_all_attributes = [
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

bank_numerical_attriubues = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

bank_attributes_label = "y"

# attributes_map = {k: v for v, k in enumerate(all_attributes)}
# test_data_path = data_path + "test.csv"
# data_path += "train.csv"

# bank_attributes_map = {k: v for v, k in enumerate(bank_all_attributes)}
# bank_test_data_path = bank_data_path + "test.csv"
# bank_data_path += "train.csv"


# six_trees = []
# sixteen_trees = []

# car_df = pd.read_csv(data_path, names=all_attributes)
# car_df.columns = all_attributes
# bank_df = pd.read_csv(bank_data_path, names=bank_all_attributes)
# bank_df.columns = bank_all_attributes

# for i in range(1, 7):
#     df = pd.read_csv(data_path, names=all_attributes)
#     ID3.attributes_label = attributes_label
#     ID3.max_depth = i
#     ID3.metric = "major_err"
#     six_trees.append(ID3.ID3_prepare(df, set(all_attributes[:-1]), "root", 0))

# for i in range(1, 17):
#     df = pd.read_csv(bank_data_path, names=bank_all_attributes)
#     ID3.attributes_label = bank_attributes_label
#     ID3.max_depth = i
#     ID3.metric = "major_err"
#     sixteen_trees.append(ID3.ID3_prepare(df, set(bank_all_attributes[:-1]), "root", 0))

# print(len(six_trees))
# print(len(sixteen_trees))


# Returns true is the value in example matches
def test_observation(tree, test_example, possible_outputs, a_map):
    if tree.data in possible_outputs and len(tree.children) == 0:
        if test_example[-1] == tree.data:
            return True
        return False
    for child in tree.children:
        if (
            test_example[a_map[tree.attribute]] == child.data
            or child.data in possible_outputs
        ):
            return test_observation(child, test_example, possible_outputs, a_map)


# Tests the resulting ID3 tree for training or test examples
def test_tree(tree, examples_path, labels, a_map, a_df, a_mode_map):
    result = []
    with open(examples_path, "r") as file:
        for line in file:
            example = line.strip().split(",")
            if "bank" in examples_path:
                example = convert_example_with_medians(example, a_df)
                example = convert_example_with_mode(example, a_mode_map)
            result.append(test_observation(tree, example, labels, a_map))
    return result.count(True) / len(result)


# Converts numerical parts of an example to binary
def convert_example_with_medians(example, b_df):
    for i, _ in enumerate(example):
        if bank_all_attributes[i] in bank_numerical_attriubues:
            med = b_df[bank_all_attributes[i]].median()
            example[i] = 1 if float(example[i]) >= med else 0

    return example


def convert_example_with_mode(example, att_modes):
    for i, _ in enumerate(example):
        if example[i] == "unknown":
            example[i] = att_modes[bank_all_attributes[i]]

    return example


def convert_df_unknown(c_df, attributes):
    for attribute in attributes:
        if is_numeric_dtype(c_df[attribute]):
            median = c_df[attribute].median()
            c_df[attribute] = c_df[attribute].apply(lambda x: 1 if x >= median else 0)

    # Comment out for using unknown as a value
    for attribute in attributes:
        mode = c_df[attribute].mode()
        if mode[0] == "unknown":
            mode = c_df[attribute].value_counts().index.tolist()[1]
        if mode[0] == "f":
            c_df[attribute] = c_df[attribute].apply(
                lambda x: mode if x == "unknown" else x
            )
        else:
            c_df[attribute] = c_df[attribute].apply(
                lambda x: mode[0] if x == "unknown" else x
            )

    return c_df


def get_mode_df_col(mode_df, attributes):
    attributes_to_mode = {}
    for attribute in attributes:
        attributes_to_mode[attribute] = mode_df[attribute].mode()[0]
    return attributes_to_mode


# train_error = []
# for tree in six_trees:
#     train_error.append(
#         test_tree(tree, data_path, ["unacc", "acc", "good", "vgood"], attributes_map)
#     )

# test_error = []
# for tree in six_trees:
#     test_error.append(
#         test_tree(
#             tree, test_data_path, ["unacc", "acc", "good", "vgood"], attributes_map
#         )
#     )

# print(f"Training error {1 - sum(train_error) / len(train_error)}")
# print(f"Testing error {1 - sum(test_error) / len(test_error)}")

# bank_df = convert_df_unknown(bank_df, bank_all_attributes)
# attributes_modes = get_mode_df_col(bank_df, bank_all_attributes[:-1])
# bank_train_error = []
# for tree in sixteen_trees:
#     bank_train_error.append(
#         test_tree(
#             tree,
#             bank_data_path,
#             ["yes", "no"],
#             bank_attributes_map,
#             bank_df,
#             attributes_modes,
#         )
#     )

# bank_test_error = []
# for tree in sixteen_trees:
#     bank_test_error.append(
#         test_tree(
#             tree,
#             bank_test_data_path,
#             ["yes", "no"],
#             bank_attributes_map,
#             bank_df,
#             attributes_modes,
#         )
#     )

# print(f"Training error {1 - sum(bank_train_error) / len(bank_train_error)}")
# print(f"Testing error {1 - sum(bank_test_error) / len(bank_test_error)}")

# df = pd.read_csv(bank_data_path, names=bank_all_attributes)
# ID3.attributes_label = bank_attributes_label
# ID3.max_depth = 5
# ID3.metric = "info_gain"
# tree_result = ID3.ID3_prepare(df, set(bank_all_attributes[:-1]), "root", 0)

# df = convert_df_unknown(df, bank_all_attributes[:-1])

# print(
#     test_tree(
#         tree_result,
#         bank_test_data_path,
#         ["yes", "no"],
#         bank_attributes_map,
#         df,
#         attributes_modes,
#     )
# )
# print(ID3.print_tree(tree_result))
