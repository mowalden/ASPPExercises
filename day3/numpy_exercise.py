import numpy as np

# a)
a = np.zeros(10)
a[4] = 5
print(f'a).\n {a}\n')

# b)
b = np.arange(10, 50)
print(f'b).\n {b}\n')

# c)
c = b[::-1]
print(f'c).\n {c}\n')
 
# d)
d = np.arange(0,9).reshape((3,3))
print(f'd).\n {d}\n')

# e)
e = np.array([1,2,0,0,4,0])
mask = e!=0
e = e[mask]
print(f'e).\n {e}\n')

# f)
f = np.random.rand(30)
mean = f.mean()
print(f'f).\n {mean}\n')

# g)
g = np.zeros((10,10))
g[0,:] = 1.
g[-1,:] = 1.
g[:,0] = 1.
g[:,-1] = 1.
print(f'g).\n {g}\n')

# h)
h = np.zeros((8,8))
h[::2,::2] = 1.
h[1::2,1::2] = 1.
print(f'h).\n {h}\n')
 
# i)
i = np.zeros((2,8))
i[0,::2] = 1.
i[1,1::2] = 1.
i = np.tile(i, (4,1))
print(f'i).\n {i}\n')

# j)
j = np.arange(11)
mask = np.logical_and(j >= 3, j <= 8)
mask = ~mask
print(f'j).\n {j[mask]}\n')

# k)
k = np.random.random(10)
k = np.sort(k)
print(f'k).\n {k}\n')

# l)
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.array_equal(A, B)
print(f'l).\n {equal}\n')

# m)
m = np.arange(10, dtype=np.int32)
print(f'm).\n {m.dtype}\t')
m = m.view(dtype=np.float32)
print(f'{m.dtype}\n')

# n)
A = np.arange(9).reshape(3,3)
B = A + 1
C = np.dot(A,B)
n = np.diag(C)
print(f'n).\n {n}\n')


