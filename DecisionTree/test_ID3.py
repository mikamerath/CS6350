import pandas as pd
import ID3

# Set this to which data you want to use (car or bank)
choice = "car"

# Set this to be the max depth allowed by the tree
max_depth = 1

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
    attributes_label = "label"
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
        "poutcome",
        "y",
    ]
    attributes_label = "y"

test_data_path = data_path + "test.csv"
data_path += "train.csv"

df = pd.read_csv(data_path)
df.columns = all_attributes

ID3.attributes_label = attributes_label
ID3.max_depth = max_depth
ID3.metric = "info_gain"

six_trees = []

for i in range(1, 7):
    six_trees.append(ID3.ID3(df, set(all_attributes[:-1]), "", i))

print(len(six_trees))


# Returns true is the value in example matches
def test_observation(tree, test_example, possible_outputs):
    if tree.data in possible_outputs:
        if test_example[-1] == tree.data:
            return True

    for child in tree.children:
        return test_observation(child, test_example, possible_outputs)
    return False


def test_tree(tree, examples_path):
    result = []
    with open(examples_path, "r") as file:
        for line in file:
            example = line.strip().split(",")
            result.append(
                test_observation(tree, example, ["unacc", "acc", "good", "vgood"])
            )
    return result.count(True) / len(result)


print(test_tree(six_trees[0], data_path))
