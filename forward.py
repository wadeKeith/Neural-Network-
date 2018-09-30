import numpy as np
from jiagou import w_1,input_layer,b_1,w_2,w_3,w_4,w_5,b_2,b_3,b_4,b_5,g

def active(x):
    for i in range(x.shape[0]):
        x[i][0] = g(x[i][0])
    return x



z_1 = np.dot(w_1, input_layer) + b_1
a_1 = active(z_1)

z_2 = np.dot(w_2,a_1) + b_2
a_2 = active(z_2)

z_3 = np.dot(w_3,a_2) + b_3
a_3 = active(z_3)

z_4 = np.dot(w_4,a_3) + b_4
a_4 = active(z_4)

z_5 =np.dot(w_5,a_4) + b_5
a_5 = active(z_5)

