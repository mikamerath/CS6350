import pandas as pd
import math

attributes_label = ""
max_depth = 6
metric = "info_gain"


class ID3Tree:
    def __init__(self, data):
        self.children = []
        self.data = data


def ID3(S, attributes, label, depth):
    if len(set(S[attributes_label])) == 1:
        return ID3Tree(S[attributes_label].iloc[0])
    if len(attributes) == 0 or depth == max_depth:
        return ID3Tree(S[attributes_label].mode().iloc[0])

    root = ID3Tree(label)
    # Last arguments should be info_gain, major_err, or gini_index
    split_attribute = get_information_gain(S, attributes, metric)
    vals = S[split_attribute].unique()
    print(split_attribute)
    for val in vals:
        subset_examples = S[S[split_attribute] == val]
        if len(subset_examples) == 0:
            root.data = ID3Tree(S[attributes_label].mode().iloc[0])
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
    print(starting_entropy)
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
            summation = sum(thing)
            if method == "info_gain":
                attribute_info_gain.append(
                    sum([-x / summation * math.log2(x / summation) for x in thing])
                    * summation
                    / total_values
                )
            elif method == "major_err":
                attribute_info_gain.append(
                    (summation - max(thing)) / float(total_values)
                )
            elif method == "gini_index":
                attribute_info_gain.append(
                    (1 - sum([(x / summation) ** 2 for x in thing]))
                    * summation
                    / total_values
                )
        # Uncomment the line below to check what the information gain is of each attribute
        print(
            f"Attribute: {attribute}\nInformation gain: {starting_entropy - sum(attribute_info_gain)}"
        )
        if best_info_gain <= starting_entropy - sum(attribute_info_gain):
            best_info_gain = starting_entropy - sum(attribute_info_gain)
            best_split = attribute
    return best_split
