import tensorflow as tf
import numpy as np
import itertools
import utils

from Functions_homework1_question1_team14 import MLP, RBFN

TEST_MLP = True
TEST_RBFN = True

if TEST_MLP:
	print("########################################################################")
	print("######################## Multi-Layer Perceptron ########################")
	print("########################################################################")

	# hparams:
	DEBUG = 5000
	TEST_SIZE = 0.3
	EPOCHS = 50000
	HIDDEN = [10, 20, 50]
	ETA = 1e-3
	RHO = [1e-3, 1e-4, 1e-5]
	SIGMA = [1, 2, 3, 4]

	# Double check hparams:
	assert DEBUG < EPOCHS, "DEBUG must be smaller than EPOCHS"
	assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"
	for idx, rho in enumerate(RHO):
		assert 1e-5 <= rho and rho <= 1e-3, "RHO[" + str(idx) + "] must be between 1e-5 and 1e-3"

	# Generate dataset:
	X, Y = utils.generate_franke_dataset()
	X_train, Y_train, X_test, Y_test = utils.split_dataset(X, Y, test_size=TEST_SIZE)

	best_test_error = float("inf")

	#utils.scatterplot_3d(X_train[:,0].tolist(), X_train[:,1].tolist(), Y_train[:,0].tolist())

	with tf.Session() as sess:
		# Hyperparameters tuning:
		for hparams in itertools.product(*[HIDDEN, SIGMA, RHO]):
			hidden_layer_size = hparams[0]
			sigma = hparams[1]
			rho = hparams[2]
			
			print("Training MLP with hparams:")
			print(" * DEBUG:", DEBUG)
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
			x_range = np.arange(0, 1, 0.01)
			y_range = np.arange(0, 1, 0.01)
			x_grid, y_grid = np.meshgrid(x_range, y_range)
			input_data = []
			for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
				input_data.append([x1, x2])
			input_data = np.array(input_data)
			z_value = np.array(mlp.predict(sess, input_data))
			z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))
			filename = "MLP_N_" + str(hidden_layer_size) + "_sigma_" + str(sigma) + "_rho_" + str(rho)
			utils.plot_3d(x_grid, y_grid, z_grid, "../images/" + filename.replace(".", ""))
			
			# Update best_mlp:
			if test_error < best_test_error:
				best_test_error = test_error
				best_hparams = hparams
				best_mlp = mlp

		print("best_test_error:", best_test_error)
		print("best_hparams:", best_hparams)
		print("best_mlp:", best_mlp.hidden_layer_size, best_mlp.sigma, best_mlp.rho)

		#y_pred = best_mlp.predict(sess, X_test)

		# Generate data to evaluate, used to plot the approximated function:
		#x_range = np.arange(0, 1, 0.01)
		#y_range = np.arange(0, 1, 0.01)
		#x_grid, y_grid = np.meshgrid(x_range, y_range)
		#input_data = []
		#for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
		#	input_data.append([x1, x2])
		#input_data = np.array(input_data)
		#z_value = np.array(best_mlp.predict(sess, input_data))
		#z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))

	# Plot the graph generated by the NN and the Franke's function:
	#utils.scatterplot_3d(X_test[:,0].tolist(), X_test[:,1].tolist(), Y_test.tolist())
	#utils.scatterplot_3d(X_test[:,0].tolist(), X_test[:,1].tolist(), y_pred)
	#utils.scatterplot_3d(input_data[:,0], input_data[:,1], z_value)
	#utils.plot_3d(x_grid, y_grid, z_grid)
	#utils.plot_franke()

if TEST_RBFN:
	print("########################################################################")
	print("#################### Radial Basis Function Network #####################")
	print("########################################################################")

	# hparams:
	DEBUG = 5000
	TEST_SIZE = 0.3
	EPOCHS = 50000
	HIDDEN = [10, 20, 50]
	ETA = 1e-3
	RHO = [1e-3, 1e-4, 1e-5]
	SIGMA = [1, 2, 3, 4]

	# Double check hparams:
	assert DEBUG < EPOCHS, "DEBUG must be smaller than EPOCHS"
	assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"
	for idx, rho in enumerate(RHO):
		assert 1e-5 <= rho and rho <= 1e-3, "RHO[" + str(idx) + "] must be between 1e-5 and 1e-3"
	for idx, sigma in enumerate(SIGMA):
		assert sigma > 0, "SIGMA[" + str(idx) + "] must be greater than 0"

	# Generate dataset:
	X, Y = utils.generate_franke_dataset()
	X_train, Y_train, X_test, Y_test = utils.split_dataset(X, Y, test_size=TEST_SIZE)

	best_test_error = float("inf")

	#utils.scatterplot_3d(X_train[:,0].tolist(), X_train[:,1].tolist(), Y_train[:,0].tolist())

	with tf.Session() as sess:
		# Hyperparameters tuning:
		for hparams in itertools.product(*[HIDDEN, SIGMA, RHO]):
			hidden_layer_size = hparams[0]
			sigma = hparams[1]
			rho = hparams[2]
			
			print("Training RBFN with hparams:")
			print(" * DEBUG:", DEBUG)
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
			x_range = np.arange(0, 1, 0.01)
			y_range = np.arange(0, 1, 0.01)
			x_grid, y_grid = np.meshgrid(x_range, y_range)
			input_data = []
			for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
				input_data.append([x1, x2])
			input_data = np.array(input_data)
			z_value = np.array(rbfn.predict(sess, input_data))
			z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))
			filename = "RBFN_N_" + str(hidden_layer_size) + "_sigma_" + str(sigma) + "_rho_" + str(rho)
			utils.plot_3d(x_grid, y_grid, z_grid, "../images/" + filename.replace(".", ""))
		
			# Update best_rbfn:
			if test_error < best_test_error:
				best_test_error = test_error
				best_hparams = hparams
				best_rbfn = rbfn

		print("best_test_error:", best_test_error)
		print("best_hparams:", best_hparams)
		print("best_rbfn:", best_rbfn.hidden_layer_size, best_rbfn.sigma, best_rbfn.rho)

		#y_pred = best_rbfn.predict(sess, X_test)

		# Generate data to evaluate, used to plot the approximated function:
		#x_range = np.arange(0, 1, 0.01)
		#y_range = np.arange(0, 1, 0.01)
		#x_grid, y_grid = np.meshgrid(x_range, y_range)
		#input_data = []
		#for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
		#	input_data.append([x1, x2])
		#input_data = np.array(input_data)
		#z_value = np.array(best_rbfn.predict(sess, input_data))
		#z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))

	# Plot the graph generated by the NN and the Franke's function:
	#utils.scatterplot_3d(X_test[:,0].tolist(), X_test[:,1].tolist(), Y_test.tolist())
	#utils.scatterplot_3d(X_test[:,0].tolist(), X_test[:,1].tolist(), y_pred)
	#utils.scatterplot_3d(input_data[:,0], input_data[:,1], z_value)
	#utils.plot_3d(x_grid, y_grid, z_grid)
	#utils.plot_franke()
