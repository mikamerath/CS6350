# Decision Tree

## Running/Testing the DecisionTree
`cd` to the DecisionTree directory and run the command `python test_ID3.py`

## Changing parameters
By default, DecisionTree uses the median to split numeric attributes and supports different depths and information gain metrics.
When testing, all depths for the ID3 tree are tested. To change the information gain metric, change the `ID3.metric` variable to
either `"info_gain"`, `"major_err"`, or `"gini_index"`