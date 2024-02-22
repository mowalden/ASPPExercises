# Program to multiply two matrices using nested loops
import random

N = 250

# NxN matrix
@profile
def create_matrix(N):
    X = []
    for i in range(N):
        X.append([random.randint(0,100) for r in range(N)])
    return X



# Nx(N+1) matrix
@profile
def create_matrix_nonsquare(N):
    Y = []
    for i in range(N):
        Y.append([random.randint(0,100) for r in range(N+1)])
    return Y

# result is Nx(N+1)
@profile
def matmul(X,Y,N):
    result = []
    for i in range(N):
        result.append([0] * (N+1))

    # iterate through rows of X
    for i in range(len(X)):
        # iterate through columns of Y
        for j in range(len(Y[0])):
            # iterate through rows of Y
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]
    return result



X = create_matrix(N)
Y = create_matrix_nonsquare(N)
result = matmul(X,Y,N)
for r in result:
    print(r)