import numpy as np
import pandas as pd

def get_one_hot_encodings(in_arr): #to get one hot encodings for 7 categorical features in input
    encoded_features = []  # List to store one-hot encoded columns

    for cat in range(7):
        feature_arr = in_arr[:, cat]
        unique_values = np.unique(feature_arr)

        # Create one-hot encodings
        one_hot = np.zeros((feature_arr.shape[0], unique_values.shape[0]))
        for i, val in enumerate(unique_values):
            one_hot[:, i] = (feature_arr == val).astype(int)

        encoded_features.append(one_hot)  # Append one-hot encoded column

    # Concatenate one-hot encoded features with the remaining features
    one_hot_encoded = np.hstack(encoded_features)
    remaining_features = in_arr[:, 7:]  # Non-categorical features (columns 7+)
    result = np.hstack((one_hot_encoded, remaining_features))

    return result


dataset = pd.read_csv(r"C:/Users/Asus/PycharmProjects/485Project/hour.csv")  #17379 x 17
sampled_data = dataset.sample(n=5000, random_state=42)
target = sampled_data["cnt"]
sampled_data.drop(["instant", "dteday", "workingday", "atemp", "casual", "registered","cnt"],axis=1,inplace=True)  #now 11 columns
input_arr=sampled_data.to_numpy()
input_arr = get_one_hot_encodings(input_arr)
target_arr=target.to_numpy()
target_arr = target_arr.reshape(-1, 1)
row_num,col_num=input_arr.shape #17379x10


tar_mean = target_arr.mean()
tar_std = target_arr.std()
target_arr = (target_arr - tar_mean) / tar_std
continuous_features = input_arr[:, -3:]  # Assuming the last 3 columns are continuous
continuous_mean = continuous_features.mean(axis=0)
continuous_std = continuous_features.std(axis=0)
input_arr[:, -3:] = (continuous_features - continuous_mean) / continuous_std


#define hyperparameters ifrst: layer num, neurons per layer and activation func loss func
layer_num = 4 # Total layers (including input, 2hidden, output)
neurons_per_layer = [input_arr.shape[1], 16, 8, 1]
  # 10 features, 2 hidden layers, 1 output
learning_rate = 0.01
epochs = 1000

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def mse(true_label, prediction):
    return np.mean((true_label - prediction) ** 2)

def mse_derivative(true_label, prediction):
    return 2 * (prediction - true_label) / len(true_label)

def initialize_weights(neurons_per_layer):
    weights = []
    biases = []
    for i in range(1, len(neurons_per_layer)):
        weights.append(np.random.randn(neurons_per_layer[i-1], neurons_per_layer[i]) * np.sqrt(2 / neurons_per_layer[i-1]))
        biases.append(np.zeros((1, neurons_per_layer[i])))
    return weights, biases


def forward_pass(X, W, B):

    hidden_results = [X]  # Input layer
    hidde_results_beforeact = []

    for w, b in zip(W, B):
        Z = np.dot(hidden_results[-1], w) + b
        hidde_results_beforeact.append(Z)
        A = relu(Z) if w is not W[-1] else Z  # Use ReLU except for the final output layer
        hidden_results.append(A)

    return hidden_results, hidde_results_beforeact


def backpropagation(hidden_results, hidde_results_beforeact, W, true_label):
    d_w = []
    d_b= []
    overall_error = mse_derivative(true_label, hidden_results[-1])  # Shape: (batch size, output neurons)

    for i in reversed(range(len(W))):
        dZ = overall_error if i == len(W) - 1 else overall_error * relu_derivative(hidde_results_beforeact[i])
        dW = np.dot(hidden_results[i].T, dZ)
        dB = np.sum(dZ, axis=0, keepdims=True)
        overall_error = np.dot(dZ, W[i].T)  # Propagate error to the previous layer

        d_w.insert(0, dW)
        d_b.insert(0, dB)

    return d_w, d_b


def gradient_descent_alg(W, B, d_w, d_b, lr):
    for i in range(len(W)):
        W[i] -= lr * d_w[i]
        B[i] -= lr * d_b[i]
    return W, B


def train(inputs, labels, W, B, epochs, lr, batch_size=16):
    sample_num = inputs.shape[0]

    for epoch in range(epochs):
        i = np.arange(sample_num)
        np.random.shuffle(i)
        inputs = inputs[i]
        labels = labels[i]

        for start in range(0, sample_num, batch_size):
            end = min(start + batch_size, sample_num)
            batched_in = inputs[start:end]
            batched_out= labels[start:end]

            # Forward pass
            hidden_results, hidde_results_beforeact = forward_pass(batched_in, W, B)

            # Backpropagation
            d_w, d_b = backpropagation(hidden_results, hidde_results_beforeact, W, batched_out)

            # Update weights
            W, B = gradient_descent_alg(weights, B, d_w, d_b, lr)

        # Monitor loss
        hidden_results, _ = forward_pass(inputs, W, B)  # Full forward pass
        loss = mse(labels, hidden_results[-1])
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return W, B

weights, biases = initialize_weights(neurons_per_layer)
weights, biases = train(input_arr, target_arr, weights, biases, epochs, learning_rate)
