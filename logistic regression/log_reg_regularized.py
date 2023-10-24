import numpy as np
import matplotlib.pyplot as plt
import math


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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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

X_train, y_train = load_data("lr_reg_data.txt")

# print("X_train:", X_train[:5])
# print("Type of X_train:", type(X_train))
# print("y_train:", y_train[:5])
# print("Type of y_train:", type(y_train))
#
# print ('The shape of X_train is: ' + str(X_train.shape))
# print ('The shape of y_train is: ' + str(y_train.shape))
# print ('We have m = %d training examples' % (len(y_train)))

# plot_data(X_train, y_train[:], pos_label='Accepted', neg_label='Rejected')
# plt.ylabel('Microchip Test 2')
# plt.xlabel('Microchip Test 1')
# plt.legend(loc='upper right')
# plt.show()

# Feature mapping function to polynomial features
def map_feature(X1, X2):
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1 ** (i - j) * (X2 ** j)))
    return np.stack(out, axis=1)


print("Original shape of data:", X_train.shape)
mapped_X = map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)

print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])

# Regularized cost function
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    m, n = X.shape
    cost_without_reg = compute_cost(X, y, w, b)
    reg_cost = 0.

    for i in range(n):
        reg_cost_i = w[i] ** 2
        reg_cost += reg_cost_i

    reg_cost = (lambda_ / (2 * m)) * reg_cost

    return cost_without_reg + reg_cost


X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost:", cost)

# Regularized gradient
def compute_gradient_reg(X, y, w, b, lambda_ = 1):
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    for i in range(n):
        dj_dw_i_reg = (lambda_ / m) * w[i]
        dj_dw[i] += dj_dw_i_reg

    return dj_db, dj_dw


X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

# print(f"dj_db {dj_db}", )
# print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 1.
lambda_ = 0.01
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b,
                                    compute_cost_reg, compute_gradient_reg,
                                    alpha, iterations, lambda_)

p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))