from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import itertools
import time
import utils


# Radial Basis Function Network:
class RBFN:
	name = "RBFN"
	
	def __init__(self, input_layer_size, hidden_layer_size, sigma, rho, eta=1e-3):
		self.hidden_layer_size = hidden_layer_size
		self.sigma = sigma
		self.rho = rho
	
		# Define computational graph:
		self.x_placeholder = tf.placeholder(tf.float32, shape=[None, input_layer_size])
		self.y_placeholder = tf.placeholder(tf.float32)

		c = tf.Variable(tf.truncated_normal([hidden_layer_size, input_layer_size]))
		v = tf.Variable(tf.truncated_normal([hidden_layer_size]))

		self.g_input = tf.expand_dims(self.x_placeholder, 1) - tf.expand_dims(c, 0) # note: tf supports broadcasting
		self.gaussian_f = tf.exp(-tf.square(tf.norm(self.g_input, axis=2)/sigma))
		self.y_p = tf.reduce_sum(tf.multiply(v, self.gaussian_f), 1)

		self.training_error = tf.reduce_mean(tf.square(self.y_p - self.y_placeholder)) / 2 + rho * (tf.reduce_sum(tf.square(c)) + tf.reduce_sum(tf.square(v)))

		# Define optimization algorithm for centers:
		self.centers_train_step = tf.train.GradientDescentOptimizer(eta).minimize(self.training_error, var_list=[c])
		
		# Define optimization algorithm for last layer weights:
		self.P = tf.placeholder(tf.float32)
		xmatrix = 2.0*(tf.matmul(self.gaussian_f, self.gaussian_f,True)/(2.0*self.P) + rho*tf.identity(float(hidden_layer_size)))
		rhs = tf.matmul(self.gaussian_f, tf.expand_dims(self.y_placeholder, 1), True) / self.P
		
		self.v_llsq = tf.matrix_solve_ls(xmatrix, rhs, fast=False)
		self.v_train_step = v.assign(tf.squeeze(self.v_llsq))
		
	
	# Train the RBFN on the dataset for a specified number of epochs using early stoppin and kfold cross-validation:
	def train(self, sess, X_train, Y_train, epochs, verbose=False, nfold=4, epsilon_err=1e-5, evaluation_step=50):
		kf = KFold(n_splits=nfold)
		last_t_err = float("inf")
		tot_epochs = 0
		time0 = time.time()
		tf.global_variables_initializer().run()
		for epoch in range(epochs*100):
			t_err = 0
			#_, _ = sess.run([self.v_llsq, self.v_train_step], feed_dict={self.x_placeholder: X_train, self.y_placeholder: Y_train, self.P: float(Y_train.shape[0])})
			#t_err, _ = sess.run([self.training_error, self.centers_train_step], feed_dict={self.x_placeholder: X_train, self.y_placeholder: Y_train})
			for trn_split, tst_split in kf.split(X_train):
				_, _ = sess.run([self.v_llsq, self.v_train_step], feed_dict={self.x_placeholder: X_train[trn_split], self.y_placeholder: Y_train[trn_split], self.P: float(Y_train[trn_split].shape[0])})
				_, _ = sess.run([self.training_error, self.centers_train_step], feed_dict={self.x_placeholder: X_train[trn_split], self.y_placeholder: Y_train[trn_split]})
				t_err = t_err + self.evaluate(sess, X_train[tst_split], Y_train[tst_split])
			t_err = t_err / nfold
			
			tot_epochs = epoch
			if verbose:
				print("Progress: %.2f%%, Training error: %.6f" % ((epoch+1)/epochs*100, t_err), end="\r")
			if epoch % evaluation_step == 0:
				if abs(t_err - last_t_err) < epsilon_err:
					break
				last_t_err = t_err
		
		training_computing_time = time.time() - time0
		
		if verbose:
			print("\nTraining time:", training_computing_time)
			print("EPOCHE SFATTE:", tot_epochs + 1) #REMOVE
		
		return training_computing_time, 0, tot_epochs

	# Evaluate the RBFN on the test set:
	def evaluate(self, sess, X_test, Y_test):
		return sess.run(self.training_error, feed_dict={self.x_placeholder: X_test, self.y_placeholder: Y_test})

	# Predict the output of the RBFN given an input:
	def predict(self, sess, X):
		return sess.run(self.y_p, feed_dict={self.x_placeholder: X})

		
		
		
def plot_approximated_function(regr, session, x_range, y_range, filename):
	x_grid, y_grid = np.meshgrid(x_range, y_range)
	input_data = []
	for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
		input_data.append([x1, x2])
	input_data = np.array(input_data)
	z_value = np.array(regr.predict(session, input_data))
	z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))
	utils.plot_3d(x_grid, y_grid, z_grid, "../images/" + filename.replace(".", ""))
	
	
	
def write_results_on_file(output, title, MSE, trainingComputingTime, numFunctionEvaluations, numGradientEvaluations):
	output.write(title)
	output.write("\nTest MSE," + "%f" % MSE)
	output.write("\nTraining computing time," + "%f" % trainingComputingTime)
	output.write("\nFunction evaluations," + "%i" % numFunctionEvaluations)
	output.write("\nGradient evaluations," + "%i\n" % numGradientEvaluations)
	