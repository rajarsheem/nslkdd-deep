import tensorflow as tf
import numpy as np
import data as d

data = np.genfromtxt('kddtrain_2class_normalized.csv', delimiter=',', skip_header=True)

# Parameters
learning_rate = 0.001
training_epochs = 8
batch_size = 1000
display_step = 1

# Network Parameters
n_hidden_1 = 30  # 1st layer num features
n_hidden_2 = 30  # 2nd layer num features
n_input = 40  # MNIST data input (img shape: 28*28)
n_classes = 5  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(_X, _weights, _biases):
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

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    # Training cycle
    for epoch in range(training_epochs):
        # total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(3000):
            batch_xs, batch_ys = d.train_batch_data(data, 40)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            if i % 250 == 0:
                print('batch', i, end=' ', flush=True)
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1))
    saver.save(sess, 'model.ckpt')
    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    a, b = d.testdata()
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: a, y: b}))