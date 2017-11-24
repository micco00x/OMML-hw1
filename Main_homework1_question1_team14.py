import tensorflow as tf
import numpy as np
import utils

from Functions_homework1_question1_team14 import MLP

# hparams:
DEBUG = 50
TEST_SIZE = 0.3
EPOCHS = 500
HIDDEN = 50
eta = 1e-3
rho = 1e-4
sigma = 4

# Double check hparams:
assert DEBUG < EPOCHS, "DEBUG must be smaller than EPOCHS"
assert TEST_SIZE <= 0.3, "TEST_SIZE must be at most 0.3"
assert 1e-5 <= rho and rho <= 1e-3, "rho must be between 1e-5 and 1e-3"

print("Training with hparams:")
print(" * DEBUG:", DEBUG)
print(" * TEST_SIZE:", TEST_SIZE)
print(" * EPOCHS:", EPOCHS)
print(" * HIDDEN:", HIDDEN)
print(" * eta:", eta)
print(" * rho:", rho)
print(" * sigma:", sigma)

# Generate dataset:
X, Y = utils.generate_franke_dataset()
X_train, Y_train, X_test, Y_test = utils.split_dataset(X, Y, test_size=TEST_SIZE)

#utils.scatterplot_3d(X_train[:,0].tolist(), X_train[:,1].tolist(), Y_train[:,0].tolist())

# Set up size of the layers:
input_layer_size = X_train.shape[1]
hidden_layer_size = HIDDEN

with tf.Session() as sess:
	mlp = MLP(input_layer_size, hidden_layer_size, sigma, rho)
	mlp.train(sess, X_train, Y_train, EPOCHS)
	print("Training error: %g" % mlp.evaluate(sess, X_train, Y_train))
	print("Test error: %g" % mlp.evaluate(sess, X_test, Y_test))
	y_pred = mlp.predict(sess, X_test)

# Define computational graph:
#x_placeholder = tf.placeholder(tf.float32, shape=[None, input_layer_size])
#y_placeholder = tf.placeholder(tf.float32)
#
#W = tf.Variable(tf.truncated_normal([input_layer_size, hidden_layer_size]))
#b = tf.Variable(tf.truncated_normal([hidden_layer_size]))
#v = tf.Variable(tf.truncated_normal([hidden_layer_size]))
#
#g_x = tf.matmul(x_placeholder, W) - b
#g_y = tf.div(1 - tf.exp(-sigma * g_x), 1 + tf.exp(-sigma * g_x)) # tanh(t/2)
#
#y_p = tf.reduce_sum(tf.multiply(v, g_y), 1)
#training_error = tf.reduce_mean(tf.square(y_p - y_placeholder)) / 2 + rho * (tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.square(b)) + tf.reduce_sum(tf.square(v)))
#
# Define optimization algorithm:
#train_step = tf.train.GradientDescentOptimizer(eta).minimize(training_error)

# Run the computational graph:
#with tf.Session() as sess:
#	tf.global_variables_initializer().run()
#	for epoch in range(EPOCHS):
#		currtrainerror, _ = sess.run([training_error, train_step], feed_dict={x_placeholder: X_train, y_placeholder: Y_train})
#		if (epoch+1) % DEBUG == 0:
#			y_pred, currtesterr = sess.run([y_p, training_error], feed_dict={x_placeholder: X_test, y_placeholder: Y_test})
#			print("Iteration %d, training error %g, test error %g" % (epoch+1, currtrainerror, currtesterr))

	# Generate data to evaluate and plot:
	x_range = np.arange(0, 1, 0.01)
	y_range = np.arange(0, 1, 0.01)
	x_grid, y_grid = np.meshgrid(x_range, y_range)
	input_data = []
	for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
		input_data.append([x1, x2])
	input_data = np.array(input_data)
	z_value = np.array(mlp.predict(sess, input_data))
	z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))

# Plot the graph generated by the NN and the Franke's function:
utils.scatterplot_3d(X_test[:,0].tolist(), X_test[:,1].tolist(), Y_test.tolist())
utils.scatterplot_3d(X_test[:,0].tolist(), X_test[:,1].tolist(), y_pred)
utils.scatterplot_3d(input_data[:,0], input_data[:,1], z_value)
utils.plot_3d(x_grid, y_grid, z_grid)
utils.plot_franke()
