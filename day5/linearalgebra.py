import scipy as sp
import numpy as np

# a)
A = np.array([[1, -2, 3],[4, 5, 6],[7, 1, 9]])

# b)
b = np.array([1, 2, 3])

# c) 
x = sp.linalg.solve(A,b)

# d)
print(f'd)\n{np.array_equal(A@x, b)}\n')

# e)
B = np.random.randint(1, 10, (3,3))
X = sp.linalg.solve(A,B)
print(f'e)\n{np.allclose(A@X,B)}\n')

# f)
eigval, eigvec = sp.linalg.eig(A)
print(f'f)\nEigenvalues: {eigval}')
print(f'Eigenvectors: {eigvec[:,0]}\n{eigvec[:,1]}\n{eigvec[:,2]}\n')

# g)
inv = sp.linalg.inv(A)
det = sp.linalg.det(A)
print(f'g)\nInverse: {inv}\nDeterminant{det}')

# h)
frob = sp.linalg.norm(A, 'fro')
first_order = sp.linalg.norm(A, 1)
second_order = sp.linalg.norm(A, 2)
print(f'h) \n Frobenius norm: {frob}')
print(f'First order: {first_order}')
print(f'Second order: {second_order}')
