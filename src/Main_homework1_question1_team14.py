import tensorflow as tf
import numpy as np
import utils

from Functions_homework1_question1_team14 import MLP, RBFN, hyperparameters_tuning, plot_approximated_function

# Debug MLP and RBFN:
TEST_MLP = True
TEST_RBFN = True

# Save figures:
SAVE_FIG = False

# Generate dataset:
TEST_SIZE = 0.3
X, Y = utils.generate_franke_dataset()
X_train, Y_train, X_test, Y_test = utils.split_dataset(X, Y, test_size=TEST_SIZE)

# Question 1 - Exercise 1: Multi-Layer Perceptron
if TEST_MLP:
	print("########################################################################")
	print("######################## Multi-Layer Perceptron ########################")
	print("########################################################################")

	# hparams:
	EPOCHS = 2
	HIDDEN = [25, 50, 75]
	ETA = 1e-3
	RHO = [1e-3, 1e-4, 1e-5]
	SIGMA = [1, 2, 3, 4]
	
	# best hparams (hidden_layer_size, sigma, rho) (found through gridsearch)
	best_hparams = (50, 2, 1e-5)

	# Double check hparams:
	assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"
	for idx, rho in enumerate(RHO):
		assert 1e-5 <= rho and rho <= 1e-3, "RHO[" + str(idx) + "] must be between 1e-5 and 1e-3"

	with tf.Session() as sess:
		## Hyperparameters tuning: (we already have the best params)
		best_mlp, best_hparams, best_test_error = hyperparameters_tuning(sess, MLP, X_train, Y_train, X_test, Y_test, HIDDEN, SIGMA, RHO, ETA, EPOCHS)
		(hidden_layer_size, sigma, rho) = (best_hparams[0], best_hparams[1], best_hparams[2])
		
		# Best RBFN
		best_mlp = MLP(X_train.shape[1], hidden_layer_size, sigma, rho, ETA)
		
		# Training
		best_mlp.train(sess, X_train, Y_train, epochs=EPOCHS)
		
		# Evalation
		best_test_error = best_mlp.evaluate(sess, X_test, Y_test)

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
	EPOCHS = 2
	HIDDEN = [25, 50, 70]
	ETA = 1e-3
	RHO = [1e-3, 1e-4, 1e-5]
	SIGMA = [0.25, 0.5, 0.75, 1]
	
	# best hparams (hidden_layer_size, sigma, rho) (found through gridsearch)
	best_hparams = (50, 0.5, 1e-5)

	# Double check hparams:
	assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"
	for idx, rho in enumerate(RHO):
		assert 1e-5 <= rho and rho <= 1e-3, "RHO[" + str(idx) + "] must be between 1e-5 and 1e-3"
	for idx, sigma in enumerate(SIGMA):
		assert sigma > 0, "SIGMA[" + str(idx) + "] must be greater than 0"

	best_test_error = float("inf")

	with tf.Session() as sess:
		## Hyperparameters tuning: (we already have the best params)
		best_rbfn, best_hparams, best_test_error = hyperparameters_tuning(sess, RBFN, X_train, Y_train, X_test, Y_test, HIDDEN, SIGMA, RHO, ETA, EPOCHS)
		(hidden_layer_size, sigma, rho) = (best_hparams[0], best_hparams[1], best_hparams[2])
		
		# Best RBFN
		best_rbfn = RBFN(X_train.shape[1], hidden_layer_size, sigma, rho, ETA)
		
		# Training
		best_rbfn.train(sess, X_train, Y_train, epochs=EPOCHS)
		
		# Evalation
		best_test_error = best_rbfn.evaluate(sess, X_test, Y_test)

		print("best_test_error:", best_test_error)
		print("best_hparams:", best_hparams)
		print("best_rbfn:", best_rbfn.hidden_layer_size, best_rbfn.sigma, best_rbfn.rho)
