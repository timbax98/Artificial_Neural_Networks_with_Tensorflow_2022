import numpy as np

a = np.eye(5,5)

print(a)

b= np.random.normal(loc=0, scale =1, size= (5,5))

print(b)

b[b>0.09] = np.square(b[b>0.09])
b[b<=0.09] = 42

print(b)
print(b[:,3])
