import numpy as np

# Generate matrix G of dimension PxN where X are the sample points,
# weights W, biases b; g is the activation function, X has dimension Pxn;
# G[r][c] = g(sum from 1 to n of W[c][i]*x[r][i] - b[c]):
def generate_MLP(X, N, g):
	P = X.shape[0]
	n = X.shape[1]
	W = np.random.normal(size=(n, N))
	b = np.random.normal(size=(N))
	
	G = g(np.matmul(X, W) - b)

	return G, W, b

# Generate matrix G of dimension PxN where X are the sample points,
# C are the centers, g is the activation function, C has dimension Nxn;
# G[r][c] = phi(norm(X[r]-C[c])):
def generate_RBFN(X, C, phi):
	P = X.shape[0]
	n = X.shape[1]
	N = C.shape[0]
	
	X = np.reshape(X, (P, 1, n))
	C = np.reshape(C, (1, N, n))
	
	G = phi(np.linalg.norm(X - C, axis=2))

	return G

	
	
def plot_approximated_function(regr, x_range, y_range, filename):
	x_grid, y_grid = np.meshgrid(x_range, y_range)
	input_data = []
	for x1, x2 in zip(np.ravel(x_grid), np.ravel(y_grid)):
		input_data.append([x1, x2])
	input_data = np.array(input_data)
	z_value = np.array(regr(input_data))
	z_grid = np.reshape(z_value, (x_grid.shape[0], x_grid.shape[1]))
	utils.plot_3d(x_grid, y_grid, z_grid, filename)