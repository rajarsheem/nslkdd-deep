import tensorflow as tf
import data as d


# Parameters
learning_rate = 0.001
training_epochs = 1
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 256  # 2nd layer num features
n_input = 40  # MNIST data input (img shape: 28*28)
n_classes = 5  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(_X, _weights, _biases):
    # layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))  # Hidden layer with RELU activation
    # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))  # Hidden layer with RELU activation
    layer_1 = tf.nn.softmax(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_2 = tf.nn.softmax(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    return tf.matmul(layer_2, _weights['out']) + _biases['out']


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,'model.ckpt')

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
a, b = d.testdata()
print(sess.run(accuracy, feed_dict={x: a, y: b}))