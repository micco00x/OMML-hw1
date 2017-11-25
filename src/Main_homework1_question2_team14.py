import numpy as np
import sklearn
from sklearn import cluster
import utils

from Functions_homework1_question2_team14 import generate_MLP, generate_RBFN

TEST_MLP = True
TEST_RBFN = True

# Question 2 - Exercise 1: Multi-Layer Perceptron
if TEST_MLP:
	print("########################################################################")
	print("######################## Multi-Layer Perceptron ########################")
	print("########################################################################")

	# hparams:
	N_EXPERIMENTS = 10000
	TEST_SIZE = 0.3
	N = 50
	sigma = 2
	rho = 1e-5

	# Double check hparams:
	assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"

	# Generate dataset:
	X, Y = utils.generate_franke_dataset()
	X_train, Y_train, X_test, Y_test = utils.split_dataset(X, Y, test_size=TEST_SIZE)

	# g activation function as specified in Q1E1:
	g = lambda t: (1-np.exp(-sigma*t))/(1+np.exp(-sigma*t))
	P = X_train.shape[0]

	best_training_error = float("inf")
	best_test_error = float("inf")

	for _ in range(N_EXPERIMENTS):
		G, W, b = generate_MLP(X_train, N, g)

		# Solve the linear system for LLSQ problem:
		v = np.linalg.solve(2*(np.matmul(G.T, G)/(2*P) + rho * np.identity(N)), np.matmul(G.T, Y_train)/P)

		# f is the MLP as specified in Q1E1:
		f = lambda x: np.sum(np.multiply(v, g(np.matmul(x, W) - b)), 1)

		# E is the error function:
		E = lambda x, y: np.mean(np.square(f(x) - y)) / 2 + rho * np.sum(np.square(v))

		training_error = E(X_train, Y_train)
		test_error = E(X_test, Y_test)

		# Update best_mlp:
		if test_error < best_test_error:
			best_training_error = training_error
			best_test_error = test_error
			best_mlp = f

	print("best_training_error:", best_training_error)
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

# Question 2 - Exercise 2: Radial Basis Function Network
if TEST_RBFN:
	print("########################################################################")
	print("#################### Radial Basis Function Network #####################")
	print("########################################################################")

	# hparams:
	N_EXPERIMENTS = 10000
	TEST_SIZE = 0.3
	N = 50
	sigma = 0.5
	rho = 1e-5

	# Double check hparams:
	assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"

	# Generate dataset:
	X, Y = utils.generate_franke_dataset()
	X_train, Y_train, X_test, Y_test = utils.split_dataset(X, Y, test_size=TEST_SIZE)

	# Select the centers randomly (TODO: use clustering):
	#C = np.take(X_train, np.random.choice(range(X_train.shape[0]), N), axis=0)
	C = sklearn.cluster.KMeans(n_clusters=N).fit(X_train).cluster_centers_

	# phi activation function as specified in Q1E2:
	phi = lambda x: np.exp(-np.square(x/sigma)) # be sure that input is of the form norm(x-c)
	P = X_train.shape[0]

	best_training_error = float("inf")
	best_test_error = float("inf")

	for _ in range(N_EXPERIMENTS):
		G = generate_RBFN(X_train, C, phi)

		# Solve the linear system for LLSQ problem:
		v = np.linalg.solve(2*(np.matmul(G.T, G)/(2*P) + rho * np.identity(N)), np.matmul(G.T, Y_train)/P)

		# f is the RBFN as specified in Q1E2:
		reshapeX = lambda x: np.reshape(x, (x.shape[0], 1, x.shape[1]))
		reshapeC = lambda c: np.reshape(c, (1, c.shape[0], c.shape[1]))
		f = lambda x: np.sum(np.multiply(v, phi(np.linalg.norm(reshapeX(x) - reshapeC(C), axis=2))), 1)

		# E is the error function:
		E = lambda x, y: np.mean(np.square(f(x) - y)) / 2 + rho * np.sum(np.square(v))

		training_error = E(X_train, Y_train)
		test_error = E(X_test, Y_test)

		# Update best_rbfn:
		if test_error < best_test_error:
			best_training_error = training_error
			best_test_error = test_error
			best_rbfn = f

	print("best_training_error:", best_training_error)
	print("best_test_error:", best_test_error)

	# Generate data to evaluate, used to plot the approximated function:
	x_range = np.arange(0, 1, 0.01)
	y_range = np.arange(0, 1, 0.01)
	x_grid, y_grid = np.meshgrid(x_range, y_range)
	input_data = []
	for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
		input_data.append([x1, x2])
	input_data = np.array(input_data)
	z_value = np.array(best_rbfn(input_data))
	z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))
	utils.plot_3d(x_grid, y_grid, z_grid, "../images/RBFN_Extreme_Learning")
