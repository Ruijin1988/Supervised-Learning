import copy, numpy as np
np.random.seed(0)

004.
# compute sigmoid nonlinearity
005.


def sigmoid(x):
    006.


output = 1 / (1 + np.exp(-x))
007.
return output
008.

009.
# convert output of sigmoid function to its derivative
010.


def sigmoid_output_to_derivative(output):
    011.


return output * (1 - output)
012.

013.

014.
# training dataset generation
015.
int2binary = {}
016.
binary_dim = 8
017.

018.
largest_number = pow(2, binary_dim)
019.
binary = np.unpackbits(
    020.
np.array([range(largest_number)], dtype=np.uint8).T, axis = 1)
021.
for i in range(largest_number):
    022.
int2binary[i] = binary[i]
023.

024.

025.
# input variables
026.
alpha = 0.1
027.
input_dim = 2
028.
hidden_dim = 16
029.
output_dim = 1
030.

031.

032.
# initialize neural network weights
033.
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
034.
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
035.
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1
036.

037.
synapse_0_update = np.zeros_like(synapse_0)
038.
synapse_1_update = np.zeros_like(synapse_1)
039.
synapse_h_update = np.zeros_like(synapse_h)
040.

041.
# training logic
042.
for j in range(10000):
    043.

044.
# generate a simple addition problem (a + b = c)
045.
a_int = np.random.randint(largest_number / 2)  # int version
046.
a = int2binary[a_int]  # binary encoding
047.

048.
b_int = np.random.randint(largest_number / 2)  # int version
049.
b = int2binary[b_int]  # binary encoding
050.

051.
# true answer
052.
c_int = a_int + b_int
053.
c = int2binary[c_int]
054.

055.
# where we'll store our best guess (binary encoded)
056.
d = np.zeros_like(c)
057.

058.
overallError = 0
059.

060.
layer_2_deltas = list()
061.
layer_1_values = list()
062.
layer_1_values.append(np.zeros(hidden_dim))
063.

064.
# moving along the positions in the binary encoding
065.
for position in range(binary_dim):
    066.

067.
# generate input and output
068.
X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
069.
y = np.array([[c[binary_dim - position - 1]]]).T
070.

071.
# hidden layer (input ~+ prev_hidden)
072.
layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
073.

074.
# output layer (new binary representation)
075.
layer_2 = sigmoid(np.dot(layer_1, synapse_1))
076.

077.
# did we miss?... if so, by how much?
078.
layer_2_error = y - layer_2
079.
layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
080.
overallError += np.abs(layer_2_error[0])
081.

082.
# decode estimate so we can print it out
083.
d[binary_dim - position - 1] = np.round(layer_2[0][0])
084.

085.
# store hidden layer so we can use it in the next timestep
086.
layer_1_values.append(copy.deepcopy(layer_1))
087.

088.
future_layer_1_delta = np.zeros(hidden_dim)
089.

090.
for position in range(binary_dim):
    091.

092.
X = np.array([[a[position], b[position]]])
093.
layer_1 = layer_1_values[-position - 1]
094.
prev_layer_1 = layer_1_values[-position - 2]
095.

096.
# error at output layer
097.
layer_2_delta = layer_2_deltas[-position - 1]
098.
# error at hidden layer
099.
layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(
    layer_1)
100.

101.
# let's update all our weights so we can try again
102.
synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
103.
synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
104.
synapse_0_update += X.T.dot(layer_1_delta)
105.

106.
future_layer_1_delta = layer_1_delta
107.

108.

109.
synapse_0 += synapse_0_update * alpha
110.
synapse_1 += synapse_1_update * alpha
111.
synapse_h += synapse_h_update * alpha
112.

113.
synapse_0_update *= 0
114.
synapse_1_update *= 0
115.
synapse_h_update *= 0
116.

117.
# print out progress
118.
if (j % 1000 == 0):
    119.
print "Error:" + str(overallError)
120.
print "Pred:" + str(d)
121.
print "True:" + str(c)
122.
out = 0
123.
for index, x in enumerate(reversed(d)):
    124.
out += x * pow(2, index)
125.
print str(a_int) + " + " + str(b_int) + " = " + str(out)
126.
print "------------"