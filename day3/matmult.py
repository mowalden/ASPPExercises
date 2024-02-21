# Program to multiply two matrices using nested loops
import random
import numpy as np

N = 250

# NxN matrix
X = np.random.randint(0, 100, (N,N))

# Nx(N+1) matrix
Y = np.random.randint(0, 100, (N,N+1))

# iterate through rows of X
result = np.matmul(X, Y)

