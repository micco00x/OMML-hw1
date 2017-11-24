import tensorflow as tf

# Multi-layer Perceptron:
class MLP:
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
