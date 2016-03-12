import tensorflow as tf
import numpy as np
import data as d


train_dataset = new_input1 = np.load('new_input.npz')['new_input1']
# Parameters

train_dataset = np.genfromtxt('train.csv', dtype=np.float32, delimiter=',')
train_labels = train_dataset[:, -1].copy()
train_labels -= 1
test_dataset = np.genfromtxt('test.csv', dtype=np.float32, delimiter=',')
test_labels = test_dataset[:, -1].copy()
test_labels -= 1

num_feature = len(train_dataset[0])
num_labels = 5


def reformat(labels):
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return labels

valid_dataset = train_dataset[-6670:, :]
valid_labels = train_labels[-6670:]
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# test_dataset, test_labels = reformat(test_dataset, test_labels)
print 'Training set', train_dataset.shape, train_labels.shape
print 'Validation set', valid_dataset.shape, valid_labels.shape
print 'Testing set', test_dataset.shape, test_labels.shape


def precision(predictions, labels):
    p = [np.argmax(x) for x in predictions]
    labs = [np.argmax(x) for x in labels]
    return 100 * precision_score(labs, p, average='macro')

batch_size = 500
graph = tf.Graph()
with graph.as_default():
    learning_rate = 0.001
    batch_size = 500
    display_step = 1

    # Network Parameters
    n_input = 2000
    n_hidden_1 = 3200
    n_hidden_2 = 800
    n_hidden_3 = 2000
    n_classes = 5

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    tf_valid_dataset = tf.constant(valid_dataset)
    dropout_rate = tf.placeholder(tf.float32)

    # Create model

    def multilayer_perceptron(X, weights, biases):
        layer_1 = tf.nn.softmax(
            tf.add(
                tf.matmul(tf.nn.dropout(_X, dropout_rate), weights['h1']),
                biases['b1'])
            )
        layer_2 = tf.nn.softmax(
            tf.add(
                tf.matmul(tf.nn.dropout(layer_1, dropout_rate), weights['h2']),
                biases['b2'])
            )
        layer_3 = tf.nn.softmax(
            tf.add(
                tf.matmul(tf.nn.dropout(layer_2, dropout_rate), weights['h2']),
                biases['b2'])
            )
        return tf.add(
            tf.matmul(tf.nn.dropout(layer_3, dropout_rate), weights['out']),
            biases['out']
            )

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)

    init = tf.initialize_all_variables()
    train_prediction = tf.nn.softmax(pred)
    valid_prediction = tf.nn.softmax(multilayer_perceptron(tf_valid_dataset))

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    sess.run(init)
    saver = tf.train.Saver()
    training_epochs = 2
    for epoch in range(training_epochs):
        for i in range(3000):
            batch_xs, batch_ys = d.train_batch_data(data, 40)
            train_feed_dict = {
                tf_train_dataset: batch_xs,
                tf_train_labels: batch_ys,
                dropout_rate: 0.7
            }

            if i % 500 == 0:
                test_feed_dict = {
                    tf_train_dataset: batch_xs,
                    tf_train_labels: batch_ys,
                    dropout_rate: 1.0
                }
                _, c = session.run(
                    [optimizer, cost], feed_dict=train_feed_dict
                )
                print(
                    "Minibatch loss at step", step, ":", c, end='\n',
                    flush=True
                )
                batch_precision = precision(
                    train_prediction.eval(feed_dict=test_feed_dict),
                    batch_labels
                )
                print(
                    "Minibatch precission at step", step, ":",
                    batch_precision,
                    end='\n',
                    flush=True
                )
                validation_precision = precision(
                    valid_prediction.eval(feed_dict=test_feed_dict),
                    valid_labels
                )
                print(
                    "Validation precission at step", step, ":",
                    validation_precision,
                    end='\n',
                    flush=True
                )
            else:
                session.run([optimizer], feed_dict=train_feed_dict)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1))
    saver.save(session, 'model.ckpt')
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    a, b = d.testdata()
    # Write code for test accuracy here
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({x: a, y: b}))
