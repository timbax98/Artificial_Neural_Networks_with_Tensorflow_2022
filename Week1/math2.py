from sympy import *
import numpy as np
import math


x = Symbol('x')
z = 1/(1 + exp(-x))
z_prime = z.diff(x)

print(z_prime)

v= Symbol('v')
a= Symbol('a')
b= Symbol('b')

f= (4*a*x**2+a)+3+(1/(1 + exp(-v)))+ ((1/(1 + exp(-b)**2)))

xprime = f.diff(x)
vprime = f.diff(v)
aprime = f.diff(a)
bprime = f.diff(b)

print(xprime)
print(vprime)
print(aprime)
print(bprime)