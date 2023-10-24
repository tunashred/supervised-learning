# I will build a logistic regression model to predict whether a student gets admitted into a university
# There is historical data from previous applicants that are to be used in the training set
# My task is to build a classification model that estimates an applicant's probability of admission based on the scores from those two exams

import numpy as np
import matplotlib.pyplot as plt
import math

def load_data(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0

    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)

X_train, y_train = load_data("lr_normal_data.txt")

# # Our data plotted
# plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")
# plt.ylabel('Exam 2 score')
# plt.xlabel('Exam 1 score')
# plt.legend(loc="upper right")
# plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b, *argv):
    m, n = X.shape
    loss_sum = 0

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = w[j] * X[i][j]
            z_wb += z_wb_ij

        z_wb += b
        f_wb = sigmoid(z_wb)
        loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
        loss_sum += loss

    total_cost = (1 / m) * loss_sum

    return total_cost

# # testing the cost function
# test_w = np.array([0.2, 0.2])
# test_b = -24.
# cost = compute_cost(X_train, y_train, test_w, test_b)
# print('{:.3f}' .format(cost))

def compute_gradient(X, y, w, b, *argv):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = X[i][j] * w[j]
            z_wb += z_wb_ij

        z_wb += b
        f_wb = sigmoid(z_wb)
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        for j in range(n):
            dj_dw_ij = (f_wb - y[i]) * X[i][j]
            dj_dw[j] += dj_dw_ij

    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

# # testing gradient
# m, n = X_train.shape
initial_w = np.array([0.2, -0.5])
initial_b = -24
# dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
# print(dj_db)
# print(dj_dw)

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    m = len(X)
    J_history = []
    w_history = []

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i < 100000:
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i% math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w_in, b_in, J_history, w_history

# # testing gradient descent
np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8
iterations = 10000
alpha = 0.001
w, b, J_history,_ = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)

def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = X[i][j] * w[j]
            z_wb += z_wb_ij

        z_wb += b
        f_wb = sigmoid(z_wb)
        p[i] = f_wb >= 0.5

    return p

# Testing the predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3
tmp_X = np.random.randn(4, 2) - 0.5

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')
p = predict(X_train, w, b)
print('Train accuracy: %f'%(np.mean(p == y_train) * 100))

