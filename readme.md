# Random Forests in Python

This is an implementation of [Random Forests](https://en.wikipedia.org/wiki/Random_forest) in Python.

Run the model using ```python3 RF.py```

## Disclaimer regarding Boston Dataset

This project was done using the Boston Housing Dataset. There are known [ethical issues](https://fairlearn.org/main/user_guide/datasets/boston_housing_data.html) with this dataset.

## Briefly, what is a Random Forest

A random forest is an ensemble machine learning model which aims to reduce overfitting by training a "forest" of many decision trees each trained on a bootstrapped dataset using randomly sampled attributes.

This implementation uses the model for regression, using the mean prediction across all trees as the predicted label. RFs can also be used for classification by means of voting.

