import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """
        
        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    # Level 1
    decision_tree_root = DecisionNode(None, None, lambda a: a[0] == 1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)
    decision_tree_root.right = DecisionNode(None, None, lambda a: a[3] == 1)

    # Level 2
    decision_tree_root.right.left = DecisionNode(None, None, lambda a: a[1] == 1)
    decision_tree_root.right.right = DecisionNode(None, None, lambda a: a[2] == 1)

    # Level 3
    decision_tree_root.right.left.left = DecisionNode(None, None, None, 0)
    decision_tree_root.right.left.right = DecisionNode(None, None, None, 1)
    decision_tree_root.right.right.left = DecisionNode(None, None, None, 0)
    decision_tree_root.right.right.right = DecisionNode(None, None, None, 1)

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for i in range(len(classifier_output)):
        if classifier_output[i] == 1 and true_labels[i] == 1:
            tp = tp + 1
        elif classifier_output[i] == 1 and true_labels[i] == 0:
            fp = fp + 1
        elif classifier_output[i] == 0 and true_labels[i] == 1:
            fn = fn + 1
        elif classifier_output[i] == 0 and true_labels[i] == 0:
            tn = tn + 1
    
    matrix = [[tp, fn], [fp, tn]]
    return matrix

def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    matrix = confusion_matrix(classifier_output, true_labels)
    tp = matrix[0][0]
    fp = matrix[1][0]
    precision = tp / (tp + fp)
    return precision

def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """
    matrix = confusion_matrix(classifier_output, true_labels)
    tp = matrix[0][0]
    fn = matrix[0][1]
    recall = tp / (tp + fn)
    return recall

def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """
    matrix = confusion_matrix(classifier_output, true_labels)
    tp = matrix[0][0]
    fn = matrix[0][1]
    fp = matrix[1][0]
    tn = matrix[1][1]

    accuracy = (tp + tn) / (tp + fn + fp + tn)
    return accuracy

def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    n = len(class_vector)
    p_1 = np.sum(class_vector) / n
    gini = p_1 * (1 - p_1) + (1 - p_1) * p_1
    return gini

def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    gini = gini_impurity(previous_classes)
    n = len(previous_classes)

    gini_list = []
    for x in current_classes:
        gini_list.append([len(x) / n, gini_impurity(x)])

    current_gini = 0
    for value in gini_list:
        current_gini = current_gini + value[0] * value[1]

    gini_gain = gini - current_gini
    return gini_gain

class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=float("inf")):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        if depth > 0: 
            
            # 1. Check for base cases - all classes or all features are the same
            if np.max(classes) == np.min(classes):
                decision_tree_root = DecisionNode(None, None, None, classes[0])
                return decision_tree_root

            if (features == features[0]).all():
                if np.sum(classes) > len(classes) / 2:
                    decision_tree_root = DecisionNode(None, None, None, 1)
                    return decision_tree_root
                else:
                    decision_tree_root = DecisionNode(None, None, None, 0)  
                    return decision_tree_root              

            # 2. For each attribute, evaluate gini gain using the average of the data as the threshold
            best_alpha = 0
            best_feature = 0
            for i in range(np.size(features,1)):
                data = features[:, i]
                avg = np.average(data)

                if np.max(data) != np.min(data):
                    l_node = data[data < avg]
                    r_node = data[data >= avg]
                    alpha = gini_gain(classes, [l_node, r_node])
                    if alpha > best_alpha:
                        best_alpha = alpha
                        best_feature = i

            # 3. Create decision node based on feature with highest alpha
            avg = np.average(features[:, best_feature])
            l_idx = np.where(features[:, best_feature] < avg)[0]
            r_idx = np.where(features[:, best_feature] >= avg)[0]

            l_features = features[l_idx,:]
            l_classes = classes[l_idx]
            r_features = features[r_idx,:]
            r_classes = classes[r_idx]

            decision_tree_root = DecisionNode(None, None, lambda a: a[best_feature] < avg)
            decision_tree_root.left = self.__build_tree__(l_features, l_classes, depth - 1)
            decision_tree_root.right = self.__build_tree__(r_features, r_classes, depth - 1)

        else:
            if np.sum(classes) > len(classes) / 2:
                decision_tree_root = DecisionNode(None, None, None, 1)
            else:
                decision_tree_root = DecisionNode(None, None, None, 0)

        return decision_tree_root

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """

        class_labels = []
        n = np.size(features,0)
        for index in range(n):
            decision = self.root.decide(features[index,:])
            class_labels.append(decision)
        return class_labels

def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    features, classes = dataset
    m = np.size(features,0)

    # First r partitions will have m // k + 1, while the remaining have m // k
    fold_list = []
    start = 0
    for i in range(k):
        step = m // k
        
        if i == 0:
            fold_train = ((features[start+step:,:], classes[start+step:]))
        else:
            train_features = np.concatenate((features[:start,:], features[start+step:,:]))
            class_features = np.concatenate((classes[:start], classes[start+step:]))
            fold_train = ((train_features, class_features))

        fold_test = ((features[start:start+step,:], classes[start:start+step]))
        fold_list.append((fold_train, fold_test))
        start = start + step
    
    return fold_list

class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        dt = DecisionTree()
        m = int(np.size(features,0))
        n = int(np.size(features,1))
        example_size = int(m * self.example_subsample_rate)
        feature_size = int(n * self.attr_subsample_rate)
        for i in range(self.num_trees):
            r_idx = np.random.randint(m, size=example_size)
            c_idx = np.random.choice(n, feature_size, replace=False)
            sample_features, sample_classes = ((features[r_idx,:][:,c_idx], classes[r_idx]))

            # Append to self.trees and reset root for loop
            self.trees.append((dt.__build_tree__(sample_features, sample_classes), c_idx))
            dt.root = None

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """
        class_labels = []
        n = np.size(features,0)

        for index in range(n):
            tree_decisions = []

            for tree in self.trees:
                root = tree[0]
                c_idx = tree[1]
                data_subset = features[:,c_idx]
                tree_decisions.append(root.decide(data_subset[index,:]))
            if np.average(tree_decisions) > 0.5:
                class_labels.append(1)
            else:
                class_labels.append(0)
        return class_labels

class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        self.trees = []
        self.num_trees = 25
        self.depth_limit = 10
        self.example_subsample_rate = 0.3
        self.attr_subsample_rate = 0.2

    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        dt = DecisionTree()
        m = int(np.size(features,0))
        n = int(np.size(features,1))
        example_size = int(m * self.example_subsample_rate)
        feature_size = int(n * self.attr_subsample_rate)
        for i in range(self.num_trees):
            r_idx = np.random.randint(m, size=example_size)
            c_idx = np.random.choice(n, feature_size, replace=False)
            sample_features, sample_classes = ((features[r_idx,:][:,c_idx], classes[r_idx]))
            
            # Append to self.trees and reset root for loop
            self.trees.append((dt.__build_tree__(sample_features, sample_classes, self.depth_limit), c_idx))
            dt.root = None

    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        class_labels = []
        n = np.size(features,0)

        for index in range(n):
            tree_decisions = []

            for tree in self.trees:
                root = tree[0]
                c_idx = tree[1]
                data_subset = features[:,c_idx]
                tree_decisions.append(root.decide(data_subset[index,:]))
            if np.average(tree_decisions) > 0.5:
                class_labels.append(1)
            else:
                class_labels.append(0)
        return class_labels

class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        data = np.add(np.multiply(data, data), data)
        return data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        index = data[:99].sum(axis=1).argmax()
        sum_max = sum(data[data[:99].sum(axis=1).argmax()])
        return sum_max, index

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        data = data[data > 0]
        data, counts = np.unique(data, return_counts=True)
        dictionary = dict(zip(data, counts))
        return dictionary.items()

def return_your_name():
    name = 'David Jaeyun Kim'
    return name
