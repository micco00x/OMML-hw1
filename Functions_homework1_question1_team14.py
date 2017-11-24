import tensorflow as tf

class MLP:
	def __init__(self, input_layer_size, hidden_layer_size, sigma, rho, eta=1e-3):
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
	def train(self, sess, X_train, Y_train, epochs):
		tf.global_variables_initializer().run()
		for epoch in range(epochs):
			sess.run([self.training_error, self.train_step], feed_dict={self.x_placeholder: X_train, self.y_placeholder: Y_train})

	# Evaluate the MLP on the test set:
	def evaluate(self, sess, X_test, Y_test):
		return sess.run(self.training_error, feed_dict={self.x_placeholder: X_test, self.y_placeholder: Y_test})

	# Predict the output of the MLP given an input:
	def predict(self, sess, X):
		return sess.run(self.y_p, feed_dict={self.x_placeholder: X})
