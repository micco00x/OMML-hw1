import tensorflow as tf
import itertools

# Multi-layer Perceptron:
class MLP:
	name = "MLP"
	
	def __init__(self, input_layer_size, hidden_layer_size, sigma, rho, eta=1e-3):
		self.hidden_layer_size = hidden_layer_size
		self.sigma = sigma
		self.rho = rho
	
		# Define computational graph:
		self.x_placeholder = tf.placeholder(tf.float32, shape=[None, input_layer_size])
		self.y_placeholder = tf.placeholder(tf.float32)

		W = tf.Variable(tf.truncated_normal([input_layer_size, hidden_layer_size]))
		b = tf.Variable(tf.truncated_normal([hidden_layer_size]))
		v = tf.Variable(tf.truncated_normal([hidden_layer_size]))

		g_x = tf.matmul(self.x_placeholder, W) - b
		g_y = tf.div(1 - tf.exp(-sigma * g_x), 1 + tf.exp(-sigma * g_x)) # tanh(t/2)

		self.y_p = tf.reduce_sum(tf.multiply(v, g_y), 1)
		self.training_error = tf.reduce_mean(tf.square(self.y_p - self.y_placeholder)) / 2 + rho * (tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(b)) + tf.reduce_sum(tf.square(v)))

		# Define optimization algorithm:
		self.train_step = tf.train.GradientDescentOptimizer(eta).minimize(self.training_error)

	# Train the MLP on the dataset for a specified number of epochs:
	def train(self, sess, X_train, Y_train, epochs, verbose=False):
		tf.global_variables_initializer().run()
		for epoch in range(epochs):
			t_err, _ = sess.run([self.training_error, self.train_step], feed_dict={self.x_placeholder: X_train, self.y_placeholder: Y_train})
			if verbose:
				print("Progress: %.2f%%, Training error: %.3f" % ((epoch+1)/epochs*100, t_err), end="\r")
		if verbose:
			print("")

	# Evaluate the MLP on the test set:
	def evaluate(self, sess, X_test, Y_test):
		return sess.run(self.training_error, feed_dict={self.x_placeholder: X_test, self.y_placeholder: Y_test})

	# Predict the output of the MLP given an input:
	def predict(self, sess, X):
		return sess.run(self.y_p, feed_dict={self.x_placeholder: X})

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

		# Define optimization algorithm:
		self.train_step = tf.train.GradientDescentOptimizer(eta).minimize(self.training_error)

	# Train the RBFN on the dataset for a specified number of epochs:
	def train(self, sess, X_train, Y_train, epochs, verbose=False):
		tf.global_variables_initializer().run()
		for epoch in range(epochs):
			t_err, _ = sess.run([self.training_error, self.train_step], feed_dict={self.x_placeholder: X_train, self.y_placeholder: Y_train})
			if verbose:
				print("Progress: %.2f%%, Training error: %.3f" % ((epoch+1)/epochs*100, t_err), end="\r")
		if verbose:
			print("")

	# Evaluate the RBFN on the test set:
	def evaluate(self, sess, X_test, Y_test):
		return sess.run(self.training_error, feed_dict={self.x_placeholder: X_test, self.y_placeholder: Y_test})

	# Predict the output of the RBFN given an input:
	def predict(self, sess, X):
		return sess.run(self.y_p, feed_dict={self.x_placeholder: X})

		


def hyperparameters_tuning(sess, Model_class, X_train, Y_train, X_test, Y_test, HIDDEN, SIGMA, RHO, ETA, EPOCHS, SAVE_FIG=False):
	
	best_test_error = float("inf")
	
	for hparams in itertools.product(*[HIDDEN, SIGMA, RHO]):
		hidden_layer_size = hparams[0]
		sigma = hparams[1]
		rho = hparams[2]
		
		print("Training", Model_class.name, "with hparams:")
		#print(" * TEST_SIZE:", TEST_SIZE)
		print(" * EPOCHS:", EPOCHS)
		print(" * HIDDEN:", hidden_layer_size)
		print(" * eta:", ETA)
		print(" * rho:", rho)
		print(" * sigma:", sigma)

		# Define the Model, train and evaluate on training and test sets:
		model = Model_class(X_train.shape[1], hidden_layer_size, sigma, rho, ETA)
		model.train(sess, X_train, Y_train, EPOCHS, True)
		training_error = model.evaluate(sess, X_train, Y_train)
		test_error = model.evaluate(sess, X_test, Y_test)
		print("Training error: %g" % training_error)
		print("Test error: %g" % test_error)
		
		# Generate data to evaluate, used to plot the approximated function:
		if SAVE_FIG:
			filename = Model_class.name + "_N_" + str(hidden_layer_size) + "_sigma_" + str(sigma) + "_rho_" + str(rho)
			plot_approximated_function(model, sess, np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), filename)

		# Update best_model:
		if test_error < best_test_error:
			best_test_error = test_error
			best_hparams = hparams
			best_model = model
	
	return best_model, best_hparams, best_test_error
		
		
		
		

def plot_approximated_function(regr, session, x_range, y_range, filename):
	x_grid, y_grid = np.meshgrid(x_range, y_range)
	input_data = []
	for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
		input_data.append([x1, x2])
	input_data = np.array(input_data)
	z_value = np.array(regr.predict(session, input_data))
	z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))
	utils.plot_3d(x_grid, y_grid, z_grid, "../images/" + filename.replace(".", ""))