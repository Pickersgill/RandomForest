import numpy as np
import math
import random
from metrics import RSS

def new_tree(data, height):
    """
    Generate a new tree with given training data up to height
    """
    
    x = data.drop(labels="medv", axis=1) # attributes
    y = np.array(data["medv"]) # labels

    root = build_tree(x, y, height)
    
    return root
    
def build_tree(xs, ys, height):
    """
    Recursive function to build tree for given set of attributes, labels and height
    """
    if height == 0: # base case
        return None

    attr, split = get_best_split(xs, ys) # determine optimal attribute and split value
    pred = np.mean(ys) # prediction label for this node

    left_split, right_split = split_on(np.array(xs[attr]), split) 
    node = Node(attr, split, pred) # create a new node

    if any(left_split) and any(right_split): # if split is not one sided, generate left and right child recursively
        node.left = build_tree(xs.loc[left_split], ys[left_split], height-1)
        node.right = build_tree(xs.loc[right_split], ys[right_split], height-1)

    return node

def get_best_split(xs, ys):
    """
    Calculate optimal split attribute and value for random attribute sample
    """
    p = len(xs.columns)
    attrs = random.sample(list(xs.columns), math.ceil(p/3)) # get random attributes to test

    best_attr = None
    best_split = None
    best_rss = None
    
    for attr in attrs:
        splits = set(np.sort(xs[attr])) # determine minimal set of test splits
        for split in splits:
            curr_rss = RSS_for_split(xs, ys, attr, split) # calculate RSS
            if best_rss is None or best_rss > curr_rss: # record lowest RSS
                best_attr = attr
                best_split = split
    
    return best_attr, best_split
  
def RSS_for_split(x, y, attr, split):
    """
    Determine RSS sum for splitting dataset (attributes=x, labels=y) on attr<=split
    """
    x_col = np.array(x[attr])
    l, r = split_on(x_col, split)
    l_rss = RSS(y[l])
    r_rss = RSS(y[r])
    total = l_rss + r_rss

    return total

def split_on(x, split):
    """
    Generate left and right indices of attribute set after splitting by given value
    """
    left = x <= split
    right = x > split
    return left, right

class Node:
    """
    Node object constituting nodes within binary decision tree.

    Each node is labelled with a prediction value and may have left and right children.
    The left and right nodes operate on a subset of the parent dataset determined by the attribute choice and split value.
    """

    def __init__(self, attr,  split, pred=None, left=None, right=None):
        self.attr = attr # attribute choice
        self.split = split # split value
        self.left = left # left child
        self.right = right # right child
        self.pred = pred # prediction label
    
    def is_leaf(self):
        return self.left is None and self.right is None

    def predict(self, row):
        """
        Predict a label for given attributes by recursively traversing the tree from this point
        """
        if self.is_leaf():
            return self.pred
        elif row[self.attr] <= self.split and self.left:
            return self.left.predict(row)
        elif row[self.attr] > self.split and self.right:
            return self.right.predict(row)
        else:
            raise Exception(f"Flaw in tree structure: \n{repr(self)}")
    
    def __str__(self):
        s = f"Node <attr={self.attr}, split={self.split}, pred={self.pred}>"
        return(s)
    
    def __repr__(self):
        s = str(self)
        if(self.left):
            s += "\n" + repr(self.left)
        if(self.right):
            s += "\n" + repr(self.right)
        return s


