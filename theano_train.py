import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib
import theano
import theano.tensor as T
import timeit
import time
from theano import pp
import data as d
from theano import config
from collections import Counter

np.random.seed(0)

class_weight = []
layer_stack, w_stack, b_stack, dw_stack, db_stack = [], [], [], [], []

num_examples = 40
nn_input_dim = 200
nn_output_dim = 5
# three layers in a conical structure
nn_hdims = [500, 100]
batch_per_epoch = 3000
num_passes = 2
n_layers = 2

epsilon = np.float32(0.06)
reg_lambda = np.float32(0.0001)


def activate(name, values):
    if name == 'sigmoid':
        return T.nnet.sigmoid(values)
    elif name == 'tanh':
        return T.tanh(values)
    elif name == 'relu':
        return T.switch(values > 0, values, 0)
    elif name == 'softmax':
        return T.nnet.softmax(values)


def add_layer(activation, dim):
    if not layer_stack:
        W = theano.shared(np.random.randn(nn_input_dim, dim).astype('float32'),
                          name='W' + str(len(layer_stack) + 1))
        b = theano.shared(np.zeros(dim).astype('float32'), name='b' + str(len(layer_stack) + 1))
        layer_stack.append(activate(activation, X.dot(W) + b))
        w_stack.append(W)
        b_stack.append(b)
    else:
        prev = layer_stack[-1]
        W = theano.shared(np.random.randn(b_stack[-1].get_value().shape[0], dim).astype('float32'),
                          name='W' + str(len(layer_stack) + 1))
        b = theano.shared(np.zeros(dim).astype('float32'), name='b' + str(len(layer_stack) + 1))
        layer_stack.append(activate(activation, prev.dot(W) + b))
        w_stack.append(W)
        b_stack.append(b)
        # print(layer_stack)


X = theano.shared(np.array(np.random.randn(200, 2000), config.floatX))
y = theano.shared(np.array(np.random.randn(200, 5), config.floatX))
c_w = theano.shared(np.array(np.random.randn(200), config.floatX))

# W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim1).astype('float32'), name='W1')
# b1 = theano.shared(np.zeros(nn_hdim1).astype('float32'), name='b1')
# W2 = theano.shared(np.random.randn(nn_hdim1, nn_hdim2).astype('float32'), name='W2')
# b2 = theano.shared(np.zeros(nn_hdim2).astype('float32'), name='b2')
# W3 = theano.shared(np.random.randn(nn_hdim2, nn_output_dim).astype('float32'), name='W3')
# b3 = theano.shared(np.zeros(nn_output_dim).astype('float32'), name='b3')
# params=[W1,b1,W2,b2,W3,b3]
params = [w_stack, b_stack]

add_layer('sigmoid', nn_hdims[0])
add_layer('sigmoid', nn_hdims[1])
add_layer('softmax', nn_output_dim)

loss_reg = 1. / num_examples * reg_lambda / 2 * (
    T.sum(T.sqr(w_stack[-3])) + T.sum(T.sqr(w_stack[-2])) + T.sum(T.sqr(w_stack[-1])))
loss = ((T.nnet.categorical_crossentropy(layer_stack[-1], y)).mean()) + loss_reg
prediction = T.argmax(layer_stack[-1], axis=1)

# dW4 = T.grad(loss, w_stack[-1])
# db4 = T.grad(loss, b_stack[-1])
# dW3 = T.grad(loss, w_stack[-2])
# db3 = T.grad(loss, b_stack[-2])
# dW2 = T.grad(loss, w_stack[-3])
# db2 = T.grad(loss, b_stack[-3])
# dW1 = T.grad(loss, w_stack[-4])
# db1 = T.grad(loss, b_stack[-4])

dw_stack = [T.grad(loss, w_stack[-n_layers+i-1]) for i in range(0, n_layers + 1)]
db_stack = [T.grad(loss, b_stack[-n_layers+i-1]) for i in range(0, n_layers + 1)]

forward_prop = theano.function([], layer_stack[-1])
calculate_loss = theano.function([], loss)
predict = theano.function([], prediction)
# u = ((w_stack[-1], w_stack[-1] - epsilon * dW4),
#      (w_stack[-2], w_stack[-2] - epsilon * dW3),
#      (w_stack[-3], w_stack[-3] - epsilon * dW2),
#      (w_stack[-4], w_stack[-4] - epsilon * dW1),
#      (b_stack[-1], b_stack[-1] - epsilon * db4),
#      (b_stack[-2], b_stack[-2] - epsilon * db3),
#      (b_stack[-3], b_stack[-3] - epsilon * db2),
#      (b_stack[-4], b_stack[-4] - epsilon * db1))

u = tuple([tuple([w_stack[-i], w_stack[-i] - epsilon * dw_stack[n_layers - i + 1]]) for i in range(1, n_layers+2)])
u += tuple([tuple([b_stack[-i], b_stack[-i] - epsilon * db_stack[n_layers - i + 1]]) for i in range(1, n_layers+2)])

gradient_step = theano.function(
    [],
    updates=u)


def build_model(num_passes=5, print_loss=False):
    np.random.seed(0)

    w_stack[-3].set_value((np.random.randn(nn_input_dim, nn_hdims[0]) / np.sqrt(nn_input_dim)).astype('float32'))
    b_stack[-3].set_value(np.zeros(nn_hdims[0]).astype('float32'))
    w_stack[-2].set_value((np.random.randn(nn_hdims[0], nn_hdims[1]) / np.sqrt(nn_hdims[0])).astype('float32'))
    b_stack[-2].set_value(np.zeros(nn_hdims[1]).astype('float32'))
    # w_stack[-1].set_value((np.random.randn(nn_hdims[1], nn_hdims[2]) / np.sqrt(nn_hdims[1])).astype('float32'))
    # b_stack[-1].set_value(np.zeros(nn_hdims[2]).astype('float32'))
    w_stack[-1].set_value((np.random.randn(nn_hdims[1], nn_output_dim) / np.sqrt(nn_hdims[1])).astype('float32'))
    b_stack[-1].set_value(np.zeros(nn_output_dim).astype('float32'))

    for i in range(0, num_passes):
        for j in range(batch_per_epoch):
            a, b, c = d.train_batch_data(40)
            X.set_value(a.astype('float32'))
            y.set_value(b.astype('float32'))
            c_w.set_value(c.astype('float32'))
            if j % 500 == 0:
                # print(debug(),end='\n',flush=True)
                print(calculate_loss(), end=' ', flush=True)
            gradient_step()

        print()
        if print_loss and i % 1 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss()))


build_model(num_passes=2, print_loss=True)

# save model
np.savez('out/model.npz', params=[[x.get_value() for x in params[0]], [x.get_value() for x in params[1]]])

a, b = d.fulldata('test')
X.set_value(a.astype('float32'))
y.set_value(b.astype('float32'))
predicted = predict()
actual = [np.argmax(x) for x in b]
print(accuracy_score(predicted, actual))
print(precision_score(actual, predicted, average='macro'))
print(confusion_matrix(actual, predicted))
