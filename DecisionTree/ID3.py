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
    info_gain = get_information_gain(S, attributes, "info_gain")


def get_information_gain(examples, attributes, method):
    total_values = len(examples.index)
    all_label_counts = examples.groupby("label").size().to_list()
    print(all_label_counts)
    starting_entropy = sum(
        [-x / total_values * math.log2(x / total_values) for x in all_label_counts]
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
            print(thing)
            summation = sum(thing)
            print(summation)
            print(total_values)
            attribute_info_gain.append(
                sum([-x / summation * math.log2(x / summation) for x in thing])
                * summation
                / total_values
            )
        print(sum(attribute_info_gain))
        print(starting_entropy - sum(attribute_info_gain))

    # counts = examples.groupby("label").count()
    # print(counts)
    # exit()
    # counts = [x for x in examples.groupby(["label"]).count().iloc[:, 0]]
    # print(counts)
    # entropy = [-x / total_values * math.log2(x / total_values) for x in counts]
    # print(entropy)
    # exit()
    # counts = examples.groupby(["buying", "label"]).size().reset_index(name="counts")
    # print(counts)


ID3(df, [], "unacc")
