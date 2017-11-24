from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Franke's function:
def franke2d(x1, x2):
	term1 = 0.75 * np.exp(-(9*x1-2)**2/4 - (9*x2-2)**2/4)
	term2 = 0.75 * np.exp(-(9*x1+1)**2/49 - (9*x2+1)/10)
	term3 = 0.5 * np.exp(-(9*x1-7)**2/4 - (9*x2-3)**2/4)
	term4 = -0.2 * np.exp(-(9*x1-4)**2 - (9*x2-7)**2)

	return term1 + term2 + term3 + term4

# Split dataset into training set and test set (test_size is a percentage):
def split_dataset(X, Y, test_size):
	X_test  = np.array(X[:int(len(X) * test_size)])
	Y_test  = np.array(Y[:int(len(Y) * test_size)])
	X_train = np.array(X[int(len(X) * test_size):])
	Y_train = np.array(Y[int(len(Y) * test_size):])

	return X_train, Y_train, X_test, Y_test

# Generate a dataset using Frank's function in the region [0,1]x[0,1]
# with a uniform noise in the range [-10^-1,10^-1]:
def generate_franke_dataset(dataset_size=100):
	np.random.seed(1764645)
	X = np.random.rand(dataset_size, 2)
	Y = franke2d(X[:,0], X[:,1]) + (np.random.rand(dataset_size) / 5 - 0.1)
	#Y = np.reshape(Y, (Y.shape[0], 1))
	
	return X, Y

# Utility function to plot Franke's function:
def plot_franke():
	# Make data.
	X = np.arange(0, 1, 0.01)
	Y = np.arange(0, 1, 0.01)
	X, Y = np.meshgrid(X, Y)
	Z = [[0] * X.shape[1] for _ in range(X.shape[0])]
	for i_idx in range(X.shape[0]):
		for j_idx in range(X.shape[1]):
			Z[i_idx][j_idx] = franke2d(X[i_idx][j_idx], Y[i_idx][j_idx])

	plot_3d(X, Y, Z)

def plot_3d(X, Y, Z, filename=None):
	fig = plt.figure(dpi=720)
	ax = fig.gca(projection='3d')

	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.set_zlim(-1.01, 1.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_xlabel("x1")
	ax.set_ylabel("x2")

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	if filename:
		plt.savefig(filename)
	else:
		plt.show()

def scatterplot_3d(X, Y, Z):
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.scatter(X, Y, Z)

	ax.set_xlabel("x1")
	ax.set_ylabel("x2")

	plt.show()
