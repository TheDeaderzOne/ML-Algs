import numpy as np
import math 

class RandomForest: 
    
    # Uses a collection of classification trees to train on random subsets of examples and features
    
    # Parameters: 
    
    # n_estimators: int
    #     # of classification trees used.
    
    # max_features: int
    #     The maximum # of features that the classification trees are allowed to use.
    
    # min_samples_split: int
    #     The minimum # of samples needed to make a split when building a tree.
    
    # min_gain: float
    #     The minimum impurity required to split the tree further. 
    
    # max_depth: int
    #     The maximum depth of a tree.
    
    
    
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2,
                 min_gain=0, max_depth=float("inf")): 
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        pass