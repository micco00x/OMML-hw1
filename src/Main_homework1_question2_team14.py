import numpy as np
import utils

from Functions_homework1_question2_team14 import generate_MLP

# hparams:
N_EXPERIMENTS = 10000
TEST_SIZE = 0.3
N = 15
sigma = 4
rho = 1e-5

# Double check hparams:
assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"

# Generate dataset:
X, Y = utils.generate_franke_dataset()
X_train, Y_train, X_test, Y_test = utils.split_dataset(X, Y, test_size=TEST_SIZE)

# g activation function as specified in Q1E1:
g = lambda t: (1-np.exp(-sigma*t))/(1+np.exp(-sigma*t))
P = X_train.shape[0]

best_test_error = float("inf")

for _ in range(N_EXPERIMENTS):
	G, W, b = generate_MLP(X_train, N, g)

	# Solve the linear system for LLSQ problem:
	v = np.linalg.solve(2/(2*P) * np.matmul(G.T, G) + rho * np.identity(N), np.matmul(G.T, Y_train)/P)

	# f is the MLP as specified in Q1E1:
	f = lambda x: np.sum(np.multiply(v, g(np.matmul(x, W) - b)), 1)

	# E is the error function:
	E = lambda x, y: np.mean(np.square(f(x) - y)) / 2 + rho * (np.sum(np.square(W)) + np.sum(np.square(b)) + np.sum(np.square(v)))

	training_error = E(X_train, Y_train)
	test_error = E(X_test, Y_test)

	# Update best_mlp:
	if test_error < best_test_error:
		best_test_error = test_error
		best_mlp = f

print("best_test_error:", best_test_error)

# Generate data to evaluate, used to plot the approximated function:
x_range = np.arange(0, 1, 0.01)
y_range = np.arange(0, 1, 0.01)
x_grid, y_grid = np.meshgrid(x_range, y_range)
input_data = []
for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
	input_data.append([x1, x2])
input_data = np.array(input_data)
z_value = np.array(best_mlp(input_data))
z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))
utils.plot_3d(x_grid, y_grid, z_grid, "../images/MLP_Extreme_Learning")
