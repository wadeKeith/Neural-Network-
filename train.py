import numpy as np



f=open("data","w")



global w_1,w_2,w_3,w_4,w_5,b_1,b_2,b_3,b_4,b_5,input_layer,y,alpa,loss
loss =0

k = 0

ninput_layer = 2
ncell_layer1 = 4
ncell_layer2 = 4
ncell_layer3 = 4
ncell_layer4 = 4
noutput_layer = 2

input = [[1],
         [2]]

alpa = -3

y = np.array([[10],
              [12]])


def a_z(x):
    y = x*(1-x)
    return y



def g(x):
    y = 1/(1+np.e**(-x))
    return y



def active(x):
    for i in range(x.shape[0]):
        x[i][0] = g(x[i][0])
    return x



input_layer = np.array(input)


input_xz = list(input_layer.shape)

input_nrows = input_xz[0]
input_ncols = input_xz[1]

w_1 = np.random.rand(ncell_layer1, input_nrows)
b_1 = np.random.rand(ncell_layer1, 1)

w_2 = np.random.rand(ncell_layer2,ncell_layer1)
b_2 = np.random.rand(ncell_layer2,1)

w_3 = np.random.rand(ncell_layer3,ncell_layer2)
b_3 = np.random.rand(ncell_layer3,1)

w_4 = np.random.rand(ncell_layer4,ncell_layer3)
b_4 =np.random.rand(ncell_layer4,1)

w_5 = np.random.rand(noutput_layer,ncell_layer4)
b_5 = np.random.rand(noutput_layer,1)







   # global w_1, w_2, w_3, w_4, w_5, b_1, b_2, b_3, b_4, b_5,input_layer,y,alpa

def iteration():
    global w_1, w_2, w_3, w_4, w_5, b_1, b_2, b_3, b_4, b_5, input_layer, y, alpa,loss
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

    if E_w5.all() !=0:
        w_5 = w_5-alpa*E_w5
    if E_b5.all() !=0:
        b_5 = b_5-alpa*E_b5
    if E_w4.all() != 0:
        w_4 = w_4-alpa*E_w4
    if E_b4.all() != 0:
        b_4 = b_4-alpa*E_b4
    if E_w3.all() != 0:
        w_3 = w_3-alpa*E_w3
    if E_b3.all() != 0:
        b_3 = b_3-alpa*E_b3
    if E_w2.all() != 0:
        w_2 = w_2-alpa*E_w2
    if E_b2.all() != 0:
        b_2 = b_2-alpa*E_b2
    if E_w1.all() != 0:
        w_1 = w_1-alpa*E_w1
    if E_b1.all() != 0:
        b_1 = b_1-alpa*E_b1






    loss = (eorr[0][0]+eorr[1][0])/2


while 1:
    if k%1000000 == 0:
        print('w1:', w_1)
        print('w2:', w_2)
        print('w3:', w_3)
        print('w4:', w_4)
        print('w5:', w_5)
        print('b1:', b_1)
        print('b2:', b_2)
        print('b3:', b_3)
        print('b4:', b_4)
        print('b5:', b_5)
        print('loss',loss)
        print('k:',k)
        f.write('w1===========\n')
        for i in w_1:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('w2===========\n')
        for i in w_2:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('w3===========\n')
        for i in w_3:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('w4===========\n')
        for i in w_4:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('w5===========\n')
        for i in w_5:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('b1===========\n')
        for i in b_1:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('b2===========\n')
        for i in b_2:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('b3===========\n')
        for i in b_3:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('b4===========\n')
        for i in b_4:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('b5===========\n')
        for i in b_5:
            i = str(i).strip('[').strip(']').replace(',', '').replace('\'', '') + '\n'  # 将其中每一个列表规范化成字符串
            f.write(i)
        f.write('=============\n')




    iteration()
    if loss<1:
        print('w1:',w_1)
        print('w2:',w_2)
        print('w3:',w_3)
        print('w4:',w_4)
        print('w5:',w_5)
        print('b1:',b_1)
        print('b2:',b_2)
        print('b3:',b_3)
        print('b4:',b_4)
        print('b5:',b_5)
        break
    k+= 1




