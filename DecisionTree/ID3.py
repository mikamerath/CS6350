import pandas as pd
import math

# Set this to which data you want to use (car or bank)
choice = "car"

# Set true to create the tree, false to test the data
train = True

# set depth to desired depth of final tree
depth = 1

if choice == "car":
    data_path = "car/"
    all_attributes = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "label",
    ]
else:
    data_path = "bank/"
    all_attributes = [
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
    ]

if train:
    data_path += "train.csv"
else:
    data_path += "test.csv"


class ID3Tree:
    def __init__(self):
        self.children = []
        self.data = None


df = pd.read_csv(data_path)
df.columns = all_attributes


def ID3(S, attributes, label):
    split_attribute = get_information_gain(S, attributes, "gini_index")
    print(split_attribute)
    return 0


def get_information_gain(examples, attributes, method):
    total_values = len(examples.index)
    all_label_counts = examples.groupby("label").size().to_list()
    best_split = ""
    best_info_gain = 0
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
    for attribute in set(all_attributes) - set(attributes) - set(["label"]):
        vals = examples[attribute].unique()
        attribute_info_gain = []
        for attribute_val in vals:
            thing = (
                examples[examples[attribute] == attribute_val]
                .groupby("label")
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


ID3(df, [], "unacc")
