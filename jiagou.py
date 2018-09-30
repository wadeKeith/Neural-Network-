import numpy as np

ninput_layer = 2
ncell_layer1 = 4
ncell_layer2 = 4
ncell_layer3 = 4
ncell_layer4 = 4
noutput_layer = 2

def g(x):
    y = 1/(1+np.e**(-x))
    return y

input = [[1],
         [2]]

input_layer = np.array(input)

#input_layer_T = np.array(input_layer).reshape(-1, 1)

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



