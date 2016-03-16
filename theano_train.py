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
# train_X, train_y = sklearn.datasets.make_moons(5000, noise=0.20)
# train_y_onehot = np.eye(2)[train_y]
train_dataset = new_input1 = np.load('out/new_input.npz')['test']


num_examples = 40
nn_input_dim = 2000
nn_output_dim = 5
nn_hdim = 1000
batch_per_epoch=3000
num_passes=2

epsilon = np.float32(0.01)
reg_lambda = np.float32(0.01)


X = theano.shared(np.array(np.random.randn(200,2000), config.floatX))
y = theano.shared(np.array(np.random.randn(200,5), config.floatX))


W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim).astype('float32'), name='W1')
b1 = theano.shared(np.zeros(nn_hdim).astype('float32'), name='b1')
W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim).astype('float32'), name='W2')
b2 = theano.shared(np.zeros(nn_output_dim).astype('float32'), name='b2')
params=[W1,b1,W2,b2]

z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
y_hat = T.nnet.softmax(z2)
loss_reg = 1./num_examples * reg_lambda/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)))
loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg
prediction = T.argmax(y_hat, axis=1)


dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)

forward_prop = theano.function([], y_hat)
calculate_loss = theano.function([], loss)
predict = theano.function([], prediction)


gradient_step = theano.function(
[],
updates=((W2, W2 - epsilon * dW2),
(W1, W1 - epsilon * dW1),
(b2, b2 - epsilon * db2),
(b1, b1 - epsilon * db1)))

def build_model(num_passes=20000, print_loss=False):
	np.random.seed(0)
	W1.set_value((np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)).astype('float32'))
	b1.set_value(np.zeros(nn_hdim).astype('float32'))
	W2.set_value((np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)).astype('float32'))
	b2.set_value(np.zeros(nn_output_dim).astype('float32'))

	for i in range(0, num_passes):
		for j in range(batch_per_epoch):
			a,b = d.train_batch_data(train_dataset, 40)
			X.set_value(a.astype('float32'))
			y.set_value(b.astype('float32'))
			if j%500==0:
				print(calculate_loss(),end=' ',flush=True)
			gradient_step()

		print()
		if print_loss and i % 1 == 0:
			print ("Loss after iteration %i: %f" %(i, calculate_loss()))

s=int(round(time.time()))
build_model(num_passes=2,print_loss=True)
e=int(round(time.time()))
# print('TIME = {0} secs'.format(e-s))

#save model
np.savez('model.npz',params=[x.get_value() for x in params])
