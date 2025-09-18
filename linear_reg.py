import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("C:/Users/Asus/PycharmProjects/485Project/hour.csv")
dataset_arr = dataset.to_numpy()
target_i = dataset.columns.get_loc("cnt")
drop_i = [dataset.columns.get_loc(col_name) for col_name in
 ["instant", "dteday", "workingday", "atemp", "casual", "registered"]]

# Define X and y
X = np.delete(dataset_arr, drop_i + [target_i], axis=1)
y = dataset_arr[:, target_i]

# Add all 1s to the 0 index column for intercept
ones_col = np.ones((17379, 1))
X = np.hstack((ones_col, X))
def one_hot_encoding(X, categorical_indices, drop_first=True):
 encoded_features = []
 num_features = X.shape[1]
 for col in range(num_features):
     if col in categorical_indices:
         unique_values = np.unique(X[:, col])
         one_hot = (X[:, col].reshape(-1, 1) == unique_values).astype(int)
         if drop_first:
             one_hot = one_hot[:, 1:]
             encoded_features.append(one_hot)
     else:
         encoded_features.append(X[:, col].reshape(-1, 1))

     return np.hstack(encoded_features)


categorical_indices = [1, 2, 3, 4, 5, 6, 7] # Indices of categorical columns
X_encoded = one_hot_encoding(X, categorical_indices)
X_final = X_encoded.astype(float) #to avoid error
X_train, X_temp, y_train, y_temp = train_test_split(X_final, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Even though the data was said to be normalized I did this to be safe.
X_train_scaled = np.hstack([X_train[:, :1], StandardScaler().fit_transform(X_train[:, 1:])])
X_val_scaled = np.hstack([X_val[:, :1], StandardScaler().transform(X_val[:, 1:])])
#X_test_scaled = np.hstack([X_test[:, :1], StandardScaler().transform(X_test[:, 1:])])

def linear_regression(X, y):
    weights = np.linalg.inv((X.T @ X) @ (X.T @ y))
    y_pred = np.dot (X, weights)
    return weights, y_pred


weights, y_train_pred = linear_regression(X_train_scaled, y_train)
y_val_pred = np.dot(X_val_scaled, weights)

train_mse = np.mean((y_train - y_train_pred) ** 2)
val_mse = np.mean((y_val - y_val_pred) ** 2)