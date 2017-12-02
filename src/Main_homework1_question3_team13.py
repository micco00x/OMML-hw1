import tensorflow as tf
import sklearn.metrics
import numpy as np
import utils

from Functions_homework1_question3_team13 import DecompositionRBFN
from Functions_homework1_question3_team13 import plot_approximated_function

# Question 3: Radial Basis Function Network with block decomposition
print("#################### Radial Basis Function Network #####################")

TEST_SIZE = 0.3
EPOCHS = 15000
ETA = 1e-3

# best hparams (hidden_layer_size, sigma, rho) (found through gridsearch)
best_hparams = (70, 0.25, 1e-5)
hidden_layer_size = best_hparams[0]
sigma = best_hparams[1]
rho = best_hparams[2]

# Double check hparams:
assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"
assert 1e-5 <= rho and rho <= 1e-3, "RHO[" + str(idx) + "] must be between 1e-5 and 1e-3"
assert sigma > 0, "SIGMA[" + str(idx) + "] must be greater than 0"

with tf.Session() as sess:
	# Dataset Generation
	X, Y = utils.generate_franke_dataset()
	X_train, Y_train, X_test, Y_test = utils.split_dataset(X, Y, test_size=TEST_SIZE)

	# RBFN creation
	rbfn = DecompositionRBFN(X_train.shape[1], hidden_layer_size, sigma, rho, ETA)

	# Training
	training_computing_time, num_function_evaluations, num_gradient_evaluations = rbfn.train(sess, X_train, Y_train, epochs=EPOCHS, verbose=True)

	# Evaluation
	mse = sklearn.metrics.mean_squared_error(rbfn.predict(sess, X_test), Y_test)

	# Data visualization
	print("rbfn:", rbfn.hidden_layer_size, rbfn.sigma, rbfn.rho)
	print("training_computing_time:", training_computing_time)
	print("num_function_evaluations:", num_function_evaluations)
	print("num_gradient_evaluations:", num_gradient_evaluations)
	print("test error (MSE):", mse)

	filename = "RBFN_BLOCK_N_" + str(hidden_layer_size) + "_sigma_" + str(sigma) + "_rho_" + str(rho)
	plot_approximated_function(rbfn, sess, np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), filename, "Two blocks RBFN")

	with open("output_homework1_team13.txt","a") as output:
		utils.write_results_on_file(output, "This is homework 1: question 3", mse, training_computing_time, num_function_evaluations, num_gradient_evaluations)
