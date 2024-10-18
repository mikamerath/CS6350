import math

from pandas.api.types import is_numeric_dtype

import numpy as np

attributes_label = ""
max_depth = 0
metric = ""


class ID3Tree:
    def __init__(self, data):
        self.children = []
        self.attribute = None
        self.data = data


def ID3_prepare(S, attributes, label, depth, forest=False, choices=2):
    S_result = S.copy(deep=True)
    for attribute in attributes:
        if is_numeric_dtype(S_result[attribute]):
            median = S_result[attribute].median()
            S_result[attribute] = S_result[attribute].apply(
                lambda x: 1 if x >= median else 0
            )

    # Comment out for using unknown as a value
    # for attribute in attributes:
    #     mode = S_result[attribute].mode()
    #     # print(mode)
    #     if mode[0] == "unknown":
    #         mode = S_result[attribute].value_counts().index.tolist()[1]
    #     if mode[0] == "f":
    #         S_result[attribute] = S_result[attribute].apply(
    #             lambda x: mode if x == "unknown" else x
    #         )
    #     else:
    #         S_result[attribute] = S_result[attribute].apply(
    #             lambda x: mode[0] if x == "unknown" else x
    #         )

    return ID3(S_result, attributes, label, depth)


def ID3(S, attributes, label, depth, forest=False, choices=2):
    if len(list(S[attributes_label].unique())) == 1:
        return ID3Tree(S[attributes_label].iloc[0])
    if len(attributes) == 0 or depth >= max_depth:
        return ID3Tree(S[attributes_label].mode().iloc[0])

    root = ID3Tree(label)
    # Last arguments should be info_gain, major_err, or gini_index
    if forest:
        split_attribute = get_information_gain(
            S, np.random.choice(attributes, choices), metric
        )
    else:
        split_attribute = get_information_gain(S, attributes, metric)
    root.attribute = split_attribute
    vals = S[split_attribute].unique()
    for val in vals:
        subset_examples = S[S[split_attribute] == val]
        if len(subset_examples.index) == 0:
            root.children.append(ID3Tree(S[attributes_label].mode().iloc[0]))
        else:
            root.children.append(
                ID3(subset_examples, attributes - {split_attribute}, val, depth + 1)
            )
    return root


def get_information_gain(examples, attributes, method):
    total_values = len(examples.index)
    all_label_counts = examples.groupby(attributes_label).size().to_list()
    best_split = ""
    best_info_gain = -1
    if method == "info_gain":
        starting_entropy = sum(
            [-x / total_values * math.log2(x / total_values) for x in all_label_counts]
        )
    elif method == "major_err":
        starting_entropy = (sum(all_label_counts) - max(all_label_counts)) / sum(
            all_label_counts
        )
    elif method == "gini_index":
        starting_entropy = 1 - sum(
            [(x / sum(all_label_counts)) ** 2 for x in all_label_counts]
        )
    for attribute in set(attributes):
        vals = examples[attribute].unique()
        attribute_info_gain = []
        for attribute_val in vals:
            thing = (
                examples[examples[attribute] == attribute_val]
                .groupby(attributes_label)
                .size()
                .to_list()
            )
            get_info_gain_of_examples(thing, method, total_values, attribute_info_gain)

        # Uncomment the line below to check what the information gain is of each attribute
        # print(
        #     f"Attribute: {attribute}\nInformation gain: {starting_entropy - sum(attribute_info_gain)}"
        # )
        if best_info_gain <= starting_entropy - sum(attribute_info_gain):
            best_info_gain = starting_entropy - sum(attribute_info_gain)
            best_split = attribute
    # print(f"Best split {best_split}, information gain {best_info_gain}")
    return best_split


def get_info_gain_of_examples(thing, method, total_values, attribute_info_gain):
    summation = sum(thing)
    if method == "info_gain":
        attribute_info_gain.append(
            sum([-x / summation * math.log2(x / summation) for x in thing])
            * summation
            / total_values
        )
    elif method == "major_err":
        attribute_info_gain.append((summation - max(thing)) / float(total_values))
    elif method == "gini_index":
        attribute_info_gain.append(
            (1 - sum([(x / summation) ** 2 for x in thing])) * summation / total_values
        )


def print_tree(tree):
    queue = []
    queue.append(tree)
    while len(queue) > 0:
        node = queue.pop(0)
        print(f"{node.data}, {node.attribute}    ")
        for child in node.children:
            queue.append(child)
