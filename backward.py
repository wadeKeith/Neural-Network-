from forward import a_5,a_4,a_3,a_2,a_1,w_5,w_3,w_4,w_2,w_1,b_5,b_4,b_3,b_2,b_1
from jiagou import input_layer
import numpy as np
alpa = 0.3


def a_z(x):
    y = x*(1-x)
    return y





y = np.array([[10],
              [12]])


#前一层梯度下降
eorr = y-a_5
a_pd5 = a_z(a_5)
z_pd5 =a_4.reshape(1,-1)
E_w5 = np.dot(eorr*a_pd5,z_pd5)
E_b5 = eorr*a_pd5
#w_5 = w_5-alpa*E_w5
#b_5 = b_5-alpa*E_b5




#第四层梯度下降
E_a4 = np.dot(np.transpose(w_5),eorr*a_pd5)
a_pd4 = a_z(a_4)
z_pd4 = a_3.reshape(1,-1)
E_w4 = np.dot(E_a4*a_pd4,z_pd4)
E_b4 = E_a4*a_pd4
#w_4 = w_4-alpa*E_w4
#b_4 = b_4-alpa*E_b4


#第三层梯度下降
E_a3 = np.dot(np.transpose(w_4),E_a4*a_pd4)
a_pd3 = a_z(a_3)
z_pd3 = a_2.reshape(1,-1)
E_w3 = np.dot(E_a3*a_pd3,z_pd3)
E_b3 = E_a3*a_pd3
#w_3 = w_3-alpa*E_w3
#b_3 = b_3-alpa*E_b3


#第二层梯度下降
E_a2 = np.dot(np.transpose(w_3),E_a3*a_pd3)
a_pd2 = a_z(a_2)
z_pd2 = a_1.reshape(1,-1)
E_w2 = np.dot(E_a2*a_pd2,z_pd2)
E_b2 = E_a2*a_pd2
#w_2 = w_2-alpa*E_w2
#b_2 = b_2-alpa*E_b2


#第一层梯度下降
E_a1 = np.dot(np.transpose(w_2),E_a2*a_pd2)
a_pd1 = a_z(a_1)
z_pd1 = input_layer.reshape(1,-1)
E_w1 = np.dot(E_a1*a_pd1,z_pd1)
E_b1 = E_a1*a_pd1
#w_1 = w_1-alpa*E_w1
#b_1 = b_1-alpa*E_b1


w_5 = w_5-alpa*E_w5
b_5 = b_5-alpa*E_b5
w_4 = w_4-alpa*E_w4
b_4 = b_4-alpa*E_b4
w_3 = w_3-alpa*E_w3
b_3 = b_3-alpa*E_b3
w_2 = w_2-alpa*E_w2
b_2 = b_2-alpa*E_b2
w_1 = w_1-alpa*E_w1
b_1 = b_1-alpa*E_b1

loss = (eorr**2)/2
#print(loss)






