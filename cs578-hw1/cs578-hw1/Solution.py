import numpy as np
class Node():
    def __init__(self, feature_index=None, threshold=None, missing=None, feature_type=None, left=None, right=None, 
                 num_l = None, num_r=None, info_gain=None, value=None):
        
        # building the decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.missing = missing
        self.feature_type = feature_type
        self.left = left
        self.right = right
        self.num_l = num_l
        self.num_r = num_r
        self.info_gain = info_gain
        
        # building the leaf node
        self.value = value

class DecisionTree():
    def __init__(self, min_samples=4, max_depth=2):
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.steps = 1
        
    def determine_feature_type(self, dataset):
        """"
        Function to determine the data type of the attribute
        Input: training data
        Output: List of data types of the features
        """
        
        feat_type = ['Categorical' if type(dataset[-1][col]) == str else 'Continuous' 
                     for col in range(np.shape(dataset)[1])]

        return feat_type
        
    def build_tree(self, dataset, feat_type, current_depth=0): 
        """
        Funtion to call the recursion and build the ID3 tree which uses information gain for splitting the data
        Input: training data, data type of features, current depth
        Output: Build the node
        """
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_data, num_features = np.shape(X)
        if num_data>=self.min_samples and current_depth<=self.max_depth:
            node_split = self.get_best_split(dataset, num_data, num_features, feat_type)
    
            if node_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(node_split["dataset_left"], feat_type, current_depth+1)
                # recur right
                right_subtree = self.build_tree(node_split["dataset_right"], feat_type, current_depth+1)
        
                return Node(node_split["feature_index"], node_split["threshold"], node_split['missing'],
                            node_split["feature_type"], left_subtree, right_subtree, node_split['num_l'],
                            node_split['num_r'], node_split["info_gain"])
        
        leaf_value = self.calculate_leaf_class(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_data, num_features, feat_type):
        """"
        Function to get the best split on the current node
        Funstion handles the missing data by assigning them to the node which maximizes information gain
        Input: train data, number of samples, number of features, data type of features
        Output: return the dictionary for the information of the best split
        """
        
        best_split = {}
        max_info_gain = -float("inf")
        n_features = [i for i in range(num_features)]
        if self.steps % 2 == 0:
            n_features = [i for i in range(num_features) if i%2 != 0]
        else:
            n_features = [i for i in range(num_features) if i%2 == 0]
        self.steps += 1
#         list_features = [i for i in range(num_features)]
#         n_features = random.sample(list_features, 4)
        for feature_index in n_features:
            feature_values = dataset[:, feature_index]
            feature_type = feat_type[feature_index]
            if feature_type == 'Categorical':
                all_thresholds = set(feature_values)
            else:
                all_thresholds = np.unique(feature_values)
            nan_flag = False
            for threshold in all_thresholds:
                if str(threshold) == 'nan':
                    nan_flag = True
                    break
            for threshold in all_thresholds:
                if str(threshold) == 'nan':
                    continue
                dataset_left, dataset_right = self.split(dataset, feature_index, feature_type, threshold)
                
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    if nan_flag:
                        dataset_nan = np.array([row for row in dataset if str(row[feature_index])=='nan'])
                        y, left_y, right_y = dataset[:, -1], np.concatenate((dataset_left, dataset_nan), axis=0)[:, -1], dataset_right[:, -1]
                        curr_info_gain_l = self.total_info_gain(y, left_y, right_y, "entropy")
                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], np.concatenate((dataset_right, dataset_nan), axis=0)[:, -1]
                        curr_info_gain_r = self.total_info_gain(y, left_y, right_y, "entropy")
                    
                        if curr_info_gain_l>curr_info_gain_r:
                            if curr_info_gain_l>max_info_gain:
                                best_split['feature_index'] = feature_index
                                best_split['threshold'] = threshold
                                best_split['missing'] = 'left'
                                best_split["feature_type"] = feature_type
                                best_split["dataset_left"] = np.concatenate((dataset_left, dataset_nan), axis=0)
                                best_split["dataset_right"] = dataset_right
                                best_split['num_l'] = len(np.concatenate((dataset_left, dataset_nan), axis=0))
                                best_split['num_r'] = len(dataset_right)
                                best_split["info_gain"] = curr_info_gain_l
                                max_info_gain = curr_info_gain_l
                        else:
                            if curr_info_gain_r>max_info_gain:
                                best_split['feature_index'] = feature_index
                                best_split['threshold'] = threshold
                                best_split['missing'] = 'right'
                                best_split["feature_type"] = feature_type
                                best_split["dataset_left"] = dataset_left
                                best_split["dataset_right"] = np.concatenate((dataset_right, dataset_nan), axis=0)
                                best_split['num_l'] = len(dataset_left)
                                best_split['num_r'] = len(np.concatenate((dataset_right, dataset_nan), axis=0))
                                best_split["info_gain"] = curr_info_gain_r
                                max_info_gain = curr_info_gain_r
    
                        
                    else:
                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                        curr_info_gain = self.total_info_gain(y, left_y, right_y, "entropy")

                        if curr_info_gain>max_info_gain:
                            best_split["feature_index"] = feature_index
                            best_split["threshold"] = threshold
                            best_split["missing"] = None
                            best_split["feature_type"] = feature_type
                            best_split["dataset_left"] = dataset_left
                            best_split["dataset_right"] = dataset_right
                            best_split['num_l'] = len(dataset_left)
                            best_split['num_r'] = len(dataset_right)
                            best_split["info_gain"] = curr_info_gain
                            max_info_gain = curr_info_gain

                        
        return best_split
    
    def split(self, dataset, feature_index, feature_type, threshold):
        """"
        Function to split the data into left node and right node
        """

        if feature_type == 'Categorical':
            dataset_left = np.array([row for row in dataset if row[feature_index]==threshold])
            dataset_right = np.array([row for row in dataset if (row[feature_index]!=threshold) and 
                                      (str(row[feature_index]) != 'nan')])
        else:
            dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold and 
                                     str(row[feature_index]) != 'nan'])
            dataset_right = np.array([row for row in dataset if row[feature_index]>threshold and
                                    str(row[feature_index] != 'nan')])
            
        return dataset_left, dataset_right
    
    def total_info_gain(self, parent, l_node, r_node, mode):
        """"
        Function to calculate the total information gain using the entropy
        """
        
        weight_l = len(l_node) / len(parent)
        weight_r = len(r_node) / len(parent)
        gain = self.calculate_entropy(parent) - (weight_l*self.calculate_entropy(l_node) + weight_r*self.calculate_entropy(r_node))
        return gain
    
    def calculate_entropy(self, y):
        """
        Function to calculate the entropy of the split
        """
        
        labels = np.unique(y)
        entropy = 0
        for label in labels:
            p_label = len(y[y == label]) / len(y)
            entropy += -p_label * np.log2(p_label)
        return entropy
    
        
    def calculate_leaf_class(self, labels):
        """
        Function to calculate the classification class of the leaf node
        """
        
        labels = list(labels)
        return max(labels, key=labels.count)
    
    
    def fit(self, X, Y):
        """
        Function to fit the training data
        """
        
        dataset = np.concatenate((X, Y), axis=1)
        feat_type = self.determine_feature_type(dataset)
        self.root = self.build_tree(dataset, feat_type)
        
    def predict(self, X):
        """
        Function to predict the classes of the test data using the build tree
        """
        
        preditions = [self.calculate_prediction(x, self.root) for x in X]
        return preditions
    
    def calculate_prediction(self, x, decision_tree):
        """
        Function to call the recursion to predict the class of the current test sample
        """
    
        
        if decision_tree.value!=None: return decision_tree.value
        feature_val = x[decision_tree.feature_index]
        if str(feature_val) == 'nan':
            if decision_tree.missing == 'left':
                return self.calculate_prediction(x, decision_tree.left)
            elif decision_tree.missing == 'right':
                return self.calculate_prediction(x, decision_tree.right)
            else:
                if decision_tree.num_l > decision_tree.num_r:
                    return self.calculate_prediction(x, decision_tree.left)
                else:
                    return self.calculate_prediction(x, decision_tree.right)
                    
        elif type(feature_val) == str:
            if feature_val == decision_tree.threshold:
                return self.calculate_prediction(x, decision_tree.left)
            else:
                return self.calculate_prediction(x, decision_tree.right)
        else:
            if feature_val<=decision_tree.threshold:
                return self.calculate_prediction(x, decision_tree.left)
            else:
                return self.calculate_prediction(x, decision_tree.right)
