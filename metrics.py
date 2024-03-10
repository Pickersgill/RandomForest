import numpy as np


def MSE(test_set, forest):
    """
    Calculate Mean Squared Error for a forest over a given test set
    """
    rss_sum = 0

    for index, row in test_set.iterrows():
        avg_pred = np.mean([tree.predict(row) for tree in forest])
        true = row.iloc[-1]
        rss_sum += (avg_pred - true) ** 2
        
    return rss_sum / len(test_set)

def RSS(ys):
    """
    Calculate Residual Sum of Squares over range of values ys, testing against mean
    """
    if len(ys) == 0:
        return 0
    yhat = np.mean(ys)
    diff = ys - yhat
    diff_sq = diff ** 2
    total = sum(diff_sq)
    return total
