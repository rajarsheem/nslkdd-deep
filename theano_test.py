import theano
import theano.tensor as T
from theano import config
import numpy as np
import data as d
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

loaded = np.load('out/model.npz')['params']

W1 = theano.shared(loaded[0])
b1 = theano.shared(loaded[1])
W2 = theano.shared(loaded[2])
b2 = theano.shared(loaded[3])
W3 = theano.shared(loaded[4])
b3 = theano.shared(loaded[5])

X = theano.shared(np.array(np.random.randn(200,2000), config.floatX))
y = theano.shared(np.array(np.random.randn(200,5), config.floatX))


z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
a2 = T.tanh(z2)
z3 = a2.dot(W3) + b3
y_hat = T.nnet.softmax(z3)
prediction = T.argmax(y_hat, axis=1)

forward_prop = theano.function([], y_hat)
predict = theano.function([], prediction)

a,b = d.fulldata('test')
X.set_value(a.astype('float32'))
y.set_value(b.astype('float32'))
predicted=predict()
actual=[np.argmax(x) for x in b]
print(accuracy_score(predicted,actual))
print(precision_score(actual, predicted,average='macro'))
print(confusion_matrix(actual, predicted))
