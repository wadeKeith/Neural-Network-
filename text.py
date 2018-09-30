import  numpy as np

a = np.array([[3],
             [4]])

def a_z(x):
    y = x*(1-x)
    return y
b = a_z(a)
print(b)