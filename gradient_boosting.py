#GRADIENT DESCENT TRY1
from itertools import combinations
from os.path import split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import xscale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from torch.masked import cumsum

dataset = pd.read_csv(r"C:\Users\alper\PycharmProjects\485projectt\hour.csv")  #17379 x 17
sampled_data = dataset.sample(n=20, random_state=42)
target = sampled_data["cnt"]
sampled_data.drop(["instant", "dteday", "workingday", "atemp", "casual", "registered","cnt"],axis=1,inplace=True)  #now 10 columns

#convert to np if u need
input_arr=sampled_data.to_numpy()
target_arr=target.to_numpy()
row_num,col_num=input_arr.shape

first_leaf_avg=(target_arr.sum())/row_num # got 189.46308763450142
residuals=target_arr-first_leaf_avg #find the first residuals
max_tree_num = 100  # Number of trees
learning_rate = 0.01
max_depth = 3

def get_best_cat_split(input_arr, previous_res):
    best_splits = []

    # first 7 cols are categorical do them first
    for c in range(7):
        feature_arr = input_arr[:, c]  # get number of unique values
        unique_val = np.unique(feature_arr)
        unique_val_number = len(unique_val)
        best_loss = float('inf')

        # groups_of_two = (2 ** (unique_val_number - 1)) - 1
        # for group in range(groups_of_two):
        unique_partitions = []
        for i in range(1, (unique_val_number // 2) + 1):
            for comb in combinations(list(unique_val), r=i):
                left = set(comb)
                right = set(list(unique_val)) - left
                partition = (frozenset(left), frozenset(right))
                if partition not in unique_partitions and (partition[1], partition[0]) not in unique_partitions:
                    unique_partitions.append(partition)

        for left, right in unique_partitions:
            # Create masks for the split
            left_mask = np.isin(feature_arr, list(left))
            right_mask = ~left_mask  # inversed

            left_res = previous_res[left_mask]
            right_res = previous_res[right_mask]

            left_loss = np.sum((left_res - np.mean(left_res)) ** 2) if len(left_res) > 0 else 0
            right_loss = np.sum((right_res - np.mean(right_res)) ** 2) if len(right_res) > 0 else 0
            total_loss = left_loss + right_loss
            if total_loss < best_loss:
                best_loss = total_loss
                best_split = left
        best_splits.append((c, best_split, best_loss))
    return best_splits

def get_best_num_split(input_arr, previous_res):
    best_num_splits=[]
    for num in range(3):
        num_feature = input_arr[:, num]
        sorted_previous_res=np.sort(previous_res)

        sorted_num_feature = np.sort(num_feature)
        unique_features=np.unique(sorted_num_feature)
        greedy_split_pts= (unique_features[:-1] + unique_features[1:]) / 2

        summed=np.sum(sorted_previous_res)
        summed_sqr=np.sum((sorted_previous_res**2))
        cumsum=np.cumsum(sorted_previous_res) #running sum of res
        cumsum_sqr=np.cumsum((sorted_previous_res**2)) #running sum of squared res

        best_split = None
        best_loss = float("inf")

        for pt in greedy_split_pts:
            # Mask for left and right groups
            left_mask = sorted_num_feature <= pt
            right_mask = ~left_mask
            # find the min sum of square residuals
            total_count = len(sorted_num_feature)
            left_count = np.sum(left_mask)
            right_count = total_count - left_count


            left_summed = cumsum[left_count - 1] if left_count > 0 else 0
            left_sum_sqr = cumsum_sqr[left_count - 1] if left_count > 0 else 0
            left_mean = left_summed / left_count if left_count > 0 else 0
            left_loss = (left_sum_sqr - 2 * left_mean * left_summed + left_count * (left_mean ** 2))

            right_summed = summed - left_summed
            right_sum_sqr = summed_sqr - left_sum_sqr
            right_mean = right_summed / right_count if right_count > 0 else 0 #cumsum/n
            right_loss = (right_sum_sqr - 2 * right_mean * right_summed + right_count * (right_mean ** 2))

            # Total loss
            total_loss = left_loss + right_loss

            if total_loss < best_loss:
                best_loss = total_loss
                best_split = pt

        best_num_splits.append((num, best_split, best_loss))
        return best_num_splits

def get_all_splits(input_arr, previous_res):

    cat_splits = get_best_cat_split(input_arr[:, :7], previous_res)
    for feature_idx, split, loss in cat_splits:
        cat_results = [(feature_idx, split, loss, 'categorical') ]

    num_splits = get_best_num_split(input_arr[:, 7:11], previous_res)  # Numerical
    for feature_idx, split, loss in num_splits:
        num_results = [(feature_idx + 7, split, loss, 'numerical')]

    splits = cat_results + num_results
    return splits

class TreeNode:
    def __init__(self, feature=None, split=None, prediction=None,feature_type=None):
        self.feature = feature
        self.split = split
        self.prediction = prediction
        self.feature_type=feature_type
        self.left = None
        self.right = None

def build_tree(features, residuals, depth=0, max_depth=3, min_samples_split=5, min_loss=1e-3):

    if depth >= max_depth or len(residuals) < min_samples_split:
        return TreeNode(prediction=np.mean(residuals))


    all_splits = get_all_splits(features, residuals)

    best_split = min(all_splits, key=lambda x: x[2])   #get the best split for that stage from the tuples get_all_splits returns, at 2nd position for every tuple
    best_feature, best_split_value, best_loss, feature_type = best_split

    if best_loss < min_loss:
        return TreeNode(prediction=np.mean(residuals))  # Create a leaf node with the mean value

    # Create node
    node = TreeNode(feature=best_feature, split=best_split_value)

    # Split the dataset
    if feature_type == 'categorical':
        left_mask = np.isin(features[:, best_feature], list(best_split_value))
        right_mask = ~left_mask
    else: #numerical then

        left_mask = features[:, best_feature] <= best_split_value
        right_mask = ~left_mask


    X_left, y_left = features[left_mask], residuals[left_mask]
    X_right, y_right = features[right_mask], residuals[right_mask]

#keep building the tree by finding the best split at each iteration
    node.left = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split, min_loss)
    node.right = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split, min_loss)

    return node

def predict_tree(node, x):
    if node.prediction is not None:
        return node.prediction  # Return the prediction if it's a leaf node

    # Handle categorical and numerical splits
    if isinstance(node.split, frozenset):  # Categorical split
        if x[node.feature] in node.split:
            return predict_tree(node.left, x)
        else:
            return predict_tree(node.right, x)
    else:  # Numerical split
        if x[node.feature] <= node.split:
            return predict_tree(node.left, x)
        else:
            return predict_tree(node.right, x)


def predict_tree_batch(node, X):
    return np.array([predict_tree(node, x) for x in X])

cumulative_predictions = np.zeros_like(target, dtype=float)  # put float type to avoid error i was getting

for tree in range(1,max_tree_num+1):
    print(residuals)
    new_tree=build_tree(input_arr,residuals,depth=0, max_depth=5, min_samples_split=10, min_loss=1e-3)
    new_pred=predict_tree_batch(new_tree,input_arr)
   # loss = np.mean((target - new_pred) ** 2)
    cumulative_predictions += learning_rate * new_pred
    residuals = target - cumulative_predictions
    loss = np.mean((target - cumulative_predictions) ** 2)
    print(f"Tree {tree}: Loss: {loss}")
