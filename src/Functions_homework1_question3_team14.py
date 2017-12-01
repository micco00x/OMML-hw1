from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import cluster
import itertools
import time
import utils

from Functions_homework1_question1_team14 import RBFN

# Radial Basis Function Network:
class DecompositionRBFN(RBFN):
	#name = "RBFN"

	def __init__(self, input_layer_size, hidden_layer_size, sigma, rho, eta=1e-3):
		super(DecompositionRBFN, self).__init__(input_layer_size, hidden_layer_size, sigma, rho, eta)

		# Redefine training error:
		self.training_error = tf.reduce_mean(tf.square(self.y_p - self.y_placeholder)) / 2 + rho * tf.reduce_sum(tf.square(self.c))

		# Redefine optimization algorithm:
		self.train_step = tf.train.GradientDescentOptimizer(eta).minimize(self.training_error, var_list=[self.c])

		# LLSQ:
		self.P = tf.placeholder(tf.float32)

		self.llsq_matrix = 2.0 * (tf.matmul(self.gaussian_f, self.gaussian_f, transpose_a=True) / (2.0 * self.P) + rho * tf.eye(hidden_layer_size))
		self.llsq_rhs = tf.matmul(self.gaussian_f, tf.expand_dims(self.y_placeholder, 1), transpose_a=True) / self.P

		self.update_v = self.v.assign(tf.squeeze(tf.matrix_solve_ls(self.llsq_matrix, self.llsq_rhs, fast=False)))

	# Train the RBFN on the dataset for a specified number of epochs using early stoppin and kfold cross-validation:
	def train(self, sess, X_train, Y_train, epochs, verbose=True, nfold=4, epsilon_err=1e-5, evaluation_step=100):
		tf.global_variables_initializer().run()
		time0 = time.time()
		last_t_err = float("inf")

		# Initial guess for centers
		cc = sklearn.cluster.KMeans(n_clusters=self.hidden_layer_size).fit(X_train).cluster_centers_
		tf.get_variable("c", shape=(cc.shape[0], cc.shape[1]), initializer=tf.constant_initializer(cc)).initializer.run()

		for epoch in range(epochs):
			sess.run(self.update_v, feed_dict={self.x_placeholder: X_train, self.y_placeholder: Y_train, self.P: X_train.shape[0]})
			t_err, _ = sess.run([self.training_error, self.train_step], feed_dict={self.x_placeholder: X_train, self.y_placeholder: Y_train})

			# kfold crossvalidation
			#for trn_split, tst_split in kf.split(X_train):
			#	_, _ = sess.run([self.v_llsq, self.v_train_step], feed_dict={self.x_placeholder: X_train[trn_split], self.y_placeholder: Y_train[trn_split], self.P: float(Y_train[trn_split].shape[0])})
			#	_, _ = sess.run([self.training_error, self.centers_train_step], feed_dict={self.x_placeholder: X_train[trn_split], self.y_placeholder: Y_train[trn_split]})
			#	t_err = t_err + self.evaluate(sess, X_train[tst_split], Y_train[tst_split])
			#t_err = t_err / nfold

			if verbose:
				print("Progress: %.2f%%, Training error: %.3f" % ((epoch+1)/epochs*100, t_err), end="\r")
			if epoch % evaluation_step == 0:
				if abs(t_err - last_t_err) < epsilon_err:
					print("\nEarly Stopping")
					break
				last_t_err = t_err
		if verbose:
			print("")

		training_computing_time = time.time() - time0
		return training_computing_time, 0, epoch


def plot_approximated_function(regr, session, x_range, y_range, filename):
	x_grid, y_grid = np.meshgrid(x_range, y_range)
	input_data = []
	for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
		input_data.append([x1, x2])
	input_data = np.array(input_data)
	z_value = np.array(regr.predict(session, input_data))
	z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))
	utils.plot_3d(x_grid, y_grid, z_grid, "../images/" + filename.replace(".", ""))
