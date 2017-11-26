import tensorflow as tf
import numpy as np
import itertools
import utils

from Functions_homework1_question1_team14 import MLP, RBFN, plot_approximated_function

# Debug MLP and RBFN:
TEST_MLP = True
TEST_RBFN = True

# Save figures:
SAVE_FIG = False

# Generate dataset:
X, Y = utils.generate_franke_dataset()
X_train, Y_train, X_test, Y_test = utils.split_dataset(X, Y, test_size=TEST_SIZE)

# Question 1 - Exercise 1: Multi-Layer Perceptron
if TEST_MLP:
	print("########################################################################")
	print("######################## Multi-Layer Perceptron ########################")
	print("########################################################################")

	# hparams:
	TEST_SIZE = 0.3
	EPOCHS = 15000
	HIDDEN = [25, 50, 75]
	ETA = 1e-3
	RHO = [1e-3, 1e-4, 1e-5]
	SIGMA = [1, 2, 3, 4]

	# Double check hparams:
	assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"
	for idx, rho in enumerate(RHO):
		assert 1e-5 <= rho and rho <= 1e-3, "RHO[" + str(idx) + "] must be between 1e-5 and 1e-3"

	best_test_error = float("inf")

	with tf.Session() as sess:
		# Hyperparameters tuning:
		for hparams in itertools.product(*[HIDDEN, SIGMA, RHO]):
			hidden_layer_size = hparams[0]
			sigma = hparams[1]
			rho = hparams[2]
			
			print("Training MLP with hparams:")
			print(" * TEST_SIZE:", TEST_SIZE)
			print(" * EPOCHS:", EPOCHS)
			print(" * HIDDEN:", hidden_layer_size)
			print(" * eta:", ETA)
			print(" * rho:", rho)
			print(" * sigma:", sigma)

			# Define MLP, train and evaluate on training and test sets:
			mlp = MLP(X_train.shape[1], hidden_layer_size, sigma, rho, ETA)
			mlp.train(sess, X_train, Y_train, EPOCHS, True)
			training_error = mlp.evaluate(sess, X_train, Y_train)
			test_error = mlp.evaluate(sess, X_test, Y_test)
			print("Training error: %g" % training_error)
			print("Test error: %g" % test_error)

			# Generate data to evaluate, used to plot the approximated function:
			if SAVE_FIG:
				filename = "MLP_N_" + str(hidden_layer_size) + "_sigma_" + str(sigma) + "_rho_" + str(rho)
				plot_approximated_function(mlp, np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), filename)
			
			# Update best_mlp:
			if test_error < best_test_error:
				best_test_error = test_error
				best_hparams = hparams
				best_mlp = mlp

		print("best_test_error:", best_test_error)
		print("best_hparams:", best_hparams)
		print("best_mlp:", best_mlp.hidden_layer_size, best_mlp.sigma, best_mlp.rho)

# Question 1 - Exercise 2: Radial Basis Function Network
if TEST_RBFN:
	print("########################################################################")
	print("#################### Radial Basis Function Network #####################")
	print("########################################################################")

	# hparams:
	TEST_SIZE = 0.3
	EPOCHS = 12500
	HIDDEN = [25, 50, 70]
	ETA = 1e-3
	RHO = [1e-3, 1e-4, 1e-5]
	SIGMA = [0.25, 0.5, 0.75, 1]

	# Double check hparams:
	assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"
	for idx, rho in enumerate(RHO):
		assert 1e-5 <= rho and rho <= 1e-3, "RHO[" + str(idx) + "] must be between 1e-5 and 1e-3"
	for idx, sigma in enumerate(SIGMA):
		assert sigma > 0, "SIGMA[" + str(idx) + "] must be greater than 0"

	best_test_error = float("inf")

	with tf.Session() as sess:
		# Hyperparameters tuning:
		for hparams in itertools.product(*[HIDDEN, SIGMA, RHO]):
			hidden_layer_size = hparams[0]
			sigma = hparams[1]
			rho = hparams[2]
			
			print("Training RBFN with hparams:")
			print(" * TEST_SIZE:", TEST_SIZE)
			print(" * EPOCHS:", EPOCHS)
			print(" * HIDDEN:", hidden_layer_size)
			print(" * eta:", ETA)
			print(" * rho:", rho)
			print(" * sigma:", sigma)

			# Define RBFN, train and evaluate on training and test sets:
			rbfn = RBFN(X_train.shape[1], hidden_layer_size, sigma, rho, ETA)
			rbfn.train(sess, X_train, Y_train, EPOCHS, True)
			training_error = rbfn.evaluate(sess, X_train, Y_train)
			test_error = rbfn.evaluate(sess, X_test, Y_test)
			print("Training error: %g" % training_error)
			print("Test error: %g" % test_error)
			
			# Generate data to evaluate, used to plot the approximated function:
			if SAVE_FIG:
				filename = "RBFN_N_" + str(hidden_layer_size) + "_sigma_" + str(sigma) + "_rho_" + str(rho)
				plot_approximated_function(rbfn, sess, np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), filename)
		
			# Update best_rbfn:
			if test_error < best_test_error:
				best_test_error = test_error
				best_hparams = hparams
				best_rbfn = rbfn

		print("best_test_error:", best_test_error)
		print("best_hparams:", best_hparams)
		print("best_rbfn:", best_rbfn.hidden_layer_size, best_rbfn.sigma, best_rbfn.rho)
