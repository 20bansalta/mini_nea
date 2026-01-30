import numpy as np


#####################
# Utility functions #
#####################

def entropy(y):
    """ 
    Calculate the entropy (base 2) for a given array of labels.

    Parameters
    ----------
    y : np.ndarray
        Array of labelss
    
    Returns
    -------
    float
        The calculated entropy
    """
    # Get the counts of different classes in the input
    _, counts = np.unique(y, return_counts=True)
    # Turn in to probability distribution (sums to 1)
    prob = counts / counts.sum()
    # We want to penalise having an even distribution - we prefer the data to be mostly in a specific class
    # We calculate the average number of bits (log2) needed to describe the class of a random sample
    return -np.sum(prob * np.log2(prob))

#####################
# Helper structures #
#####################

class NodeInterface:
    """
    An base interface class for nodes of a decision tree.
    """

    def is_leaf(self):
        raise NotImplementedError("Override not provided for Node.is_leaf()")

    def __str__(self, depth=0):
        raise NotImplementedError("Override not provided for Node.__str__()")

class DecisionNode(NodeInterface):
    """
    A node which splits by a threshold, and has a left/right child node.
    It inherits for and implements the abstract methods of NodeInterface.
    """
    # Constructor
    def __init__(self, feature, threshold, left, right):
        self.__feature = feature     # The index of the feature we split our data on
        self.__threshold = threshold # If under threshold, go to left node, otherwise right node
        self.__left = left           # The left node
        self.__right = right         # The right node

    # Overrides
    def is_leaf(self): return False

    def __str__(self, depth=0):
        indent = "  " * depth
        formatted_string = f"{indent}DecisionNode(feature={self.__feature}, threshold={self.__threshold:.4f})\n" \
                           + self.__left.__str__(depth + 1) + "\n" \
                           + self.__right.__str__(depth + 1)
        
        return formatted_string

    # Getters
    def get_feature(self): return self.__feature
    def get_threshold(self): return self.__threshold
    def get_left(self): return self.__left
    def get_right(self): return self.__right

class LeafNode(NodeInterface):
    def __init__(self, value):
        self.__value = value

    # Overrides
    def is_leaf(self): return True

    def __str__(self, depth=0):
        indent = " " * depth
        return f"LeafNode(value={self.__value})"

    # Getters
    def get_value(self): return self.__value


class BestSplit:
    """ 
    A simple data structure to pass information about the optimal split between methods.
    
    Attributes
    ----------
    feature : int
        The feature index (column) to split on

    threshold : float
        The threshold used to decide whether to traverse into the left/right subtree

    gain : float    
        The reduction in entropy from making this split - cached to avoid recomputation

    masks : tuple[np.ndarray, np.ndarray]
        A pair of Boolean arrays to filter data into the left/right subtree - cached to avoid recomputation
    
    y_split : tuple[np.ndarray, np.ndarray]
        The split labels
    """

    def __init__(self, min_gain):
        self.feature = None
        self.threshold = None
        self.gain = min_gain
        self.masks = (None, None)
        self.y_split = (None, None)

######################
# Main class         #
######################

class DecisionTree:
    """
    The class that stores Nodes and provide an interface for prediction.
    """

    def __init__(self, x, y, max_depth=None, min_gain=1e-7):
        self.__root = DecisionTree.__build(x, y, 0, max_depth, min_gain)

    @staticmethod
    def __best_split(x, y, min_gain):
        n_samples, n_features = x.shape

        best = BestSplit()
        old_entropy = entropy(y)

        for feature in range(n_features):
            values = np.sort(np.unique(x[:, feature]))
            # Average between adjacent values
            thresholds = (values[:-1] + values[1:]) / 2

            for t in thresholds:
                left_mask = x[:, feature] <= t
                right_mask = x[:, feature] > t

                # If none of the data is contained in one of the masks, then this is a useless threshold
                if not any(left_mask) or not any(right_mask):
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                # .mean() on a boolean mask gets what fraction of the values are put into mask
                # We use it as our weight (left_mask.mean() + right_mask.mean() == 1)
                new_entropy = (left_mask.mean() * entropy(y_left) +
                               right_mask.mean() * entropy(y_right))

                # The gain is just the reduction in entropy
                gain = old_entropy - new_entropy
                if gain > best['gain']:
                    best.feature  = feature
                    best.threshold = t
                    best.gain = gain
                    best.masks = (left_mask, right_mask)
                    best.y_split = (y_left, y_right)
        
        return best

    @staticmethod
    def __build(x, y, depth, max_depth, min_gain):
        # If we've gone to our maximum depth, or we've only got 1 remaining class
        classes, counts = np.unique(y, return_counts=True)
        if (max_depth is not None and depth >= max_depth) or len(classes) == 1:
            # We use the highest count to choose our value at the end
            return LeafNode(classes[np.argmax(counts)])
        
        best = DecisionTree.__best_split(x, y, min_gain)
        # No useful splits
        if best['gain'] == min_gain:
            return LeafNode(classes[np.argmax(counts)])

        left_mask, right_mask = best['masks']
        y_left, y_right = best['y_split']
            
        left  = DecisionTree.__build(x[left_mask],  y_left,  depth + 1, max_depth, min_gain)
        right = DecisionTree.__build(x[right_mask], y_right, depth + 1, max_depth, min_gain)

        return DecisionNode(
            best['feature'],
            best['threshold'],
            left,
            right
        )

    @staticmethod
    def __predict_helper(node, x):
        if node.is_leaf():
            return node.value
            
        return DecisionTree._predict_helper(
            node.get_left() if x[node.feature] <= node.get_threshold() else node.get_right(),
            x
        )
    
    def predict(self, x):
        """
        Predicts labels for a given table of feature records

        Parameters
        ----------
        x : np.ndarray
            Table of feature records
        
        Returns
        -------
        np.ndarray
            The predicted labels
        """
        return np.array([DecisionTree.__predict_helper(self.__root, xr) for xr in x])