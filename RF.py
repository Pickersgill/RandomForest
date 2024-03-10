import random
import sys
import numpy as np
import pandas as pd
from tree import new_tree
import metrics

def make_bts(data):
    """
    Generate a bootstrapped training set from a pandas df

    Returns a pandas df
    """
    n = len(data)
    b_inds = np.random.randint(n, size=n)
    return(data.iloc[b_inds])

def train_forest(size, height, train_set):
    """
    Train a random forest of given size (number of trees), height (depth of trees) and training data

    Returns a list of root nodes
    """
    forest = []
    bar_char = "█"
    sys.stdout.write("Generating Forest...\n")
    
    for i in range(size):
        complete = int((i + 1) / size * 100) # calculate completion percentage
        sys.stdout.write(bar_char * complete + "\b" * complete) # print loading bar, return to start of line
        sys.stdout.flush()
        bootstrap = make_bts(train_set)
        forest.append(new_tree(bootstrap, height))
    print()

    return forest

def get_sets(data, test_train_ratio):
    """
    Split Pandas df into test and train set according to given ratio

    Returns a tuple (train, test) of dataframes
    """
    rows = list(range(0, len(data))) # generate list of all indices
    random.shuffle(rows) # shuffle indices
    split = int(test_train_ratio * len(data)) 
    train_set = boston.loc[rows[split:]] 
    test_set = boston.loc[rows[:split]]
    return train_set, test_set

def test_range(sizes, heights, train, test):
    """
    Test a range of forest sizes and heights and return a matrix of resulting MSEs

    Index i, j contains MSE for the ith value of sizes and jth value of heights
    """
    results = np.array((len(sizes), len(heights)))

    for i, b in enumerate(sizes):
        for j, h in enumerate(heights):
            forest = train_forest(b, h, train)
            mse = metrics.MSE(test, forest)
            print("b: %d, h: %d, MSE: %f" % (b, h, mse))
            results[i][j] = mse
    
    return results

if __name__ == "__main__":
    #random.seed(015)
    #np.random.seed(015)

    boston = pd.read_csv("./boston.csv")
    B = 20 # forest size
    h = 5 # tree height
    tt_ratio = 0.5 # 50:50 train:test split
    train_set, test_set = get_sets(boston, tt_ratio)

    forest = train_forest(B, h, train_set)

    train_mse = metrics.MSE(train_set, forest)
    test_mse = metrics.MSE(test_set, forest)
    print("Training MSE: ", train_mse)
    print("Testing MSE: ", test_mse)

    print(repr(forest[0]))

