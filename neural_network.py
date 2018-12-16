import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Training Models Using TensorFlows High Level API
# ------------------------------------------------------------------------------

# The following code will explore two of tensorflows high level API's. One is called Layers and the other is known as Keras. To demonstrate, the MNIST data set will be used and a neural network will be designed to tackle the same problem. First, the data set is loaded

import os
import struct


def load_mnist(path, kind='train'):

    labels_path = os.path.join(path, '{}-labels.idx1-ubyte'.format(kind))
    images_path = os.path.join(path, '{}-images.idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))

        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

        images = ((images / 255.) - .5) * 2

    return images, labels


X_train, y_train = load_mnist('', kind='train')

X_test, y_test = load_mnist('', kind='t10k')

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val

del X_train, X_test

# now that the data is loaded, the model can be built.

n_features = X_train_centered.shape[1]
n_classes = np.unique(y_train).shape[0]

random_seed = 123
np.random.seed(123)

g = tf.Graph()

with g.as_default():

    tf.set_random_seed(random_seed)

    # two placeholder are then defined

    tf_x = tf.placeholder(dtype=tf.float32, shape=(
        None, n_features), name='tf_x')
    tf_y = tf.placeholder(dtype=tf.int32, shape=None, name='tf_y')

    y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)

    # the first hidden layer is then defined

    h1 = tf.layers.dense(inputs=tf_x, units=50,
                         activation=tf.tanh, name='layer1')

    # and the second hidden layer

    h2 = tf.layers.dense(inputs=h1, units=50,
                         activation=tf.tanh, name='layer2')

    # and finally the output layer

    logits = tf.layers.dense(
        inputs=h2, units=10, activation=None, name='layer3')

    predictions = {'classes': tf.argmax(logits, axis=1, name='predicted-classes'),
                   'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

    # next, the cost function is defined along with an operator to initialize the model parameters as well as an optimization operator

with g.as_default():

    cost = tf.losses.softmax_cross_entropy(
        onehot_labels=y_onehot, logits=logits)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    train_op = optimizer.minimize(loss=cost)

    init_op = tf.global_variables_initializer()

    # before the model is trained, however, one needs a way to generate batches of data


def create_batch_generator(X, y, batch_size=128, shuffle=False):

    X_copy = np.array(X)
    y_copy = np.array(y)

    if shuffle:
        data = np.column_stack((X_copy, y_copy))
        np.random.shuffle(data)
        X_copy = data[:, :-1]
        y_copy = data[:, -1].astype(int)

    for i in range(0, X.shape[0], batch_size):
        yield (X_copy[i:i + batch_size, :], y_copy[i:i + batch_size])

# next, the tensorflow session is created


sess = tf.Session(graph=g)

# the variable initializer is then run

sess.run(init_op)

# 50 epochs of training data are then interated over

for epoch in range(50):

    training_cost = []

    # a batch generator object is then created

    batch_generator = create_batch_generator(
        X_train_centered, y_train, batch_size=64)

    for batch_X, batch_y in batch_generator:

        feed = {tf_x: batch_X, tf_y: batch_y}
        _, batch_cost = sess.run([train_op, cost], feed_dict=feed)
        training_cost.append(batch_cost)

        print('Epoch: {}\nAverage Training Cost: {}'.format(epoch + 1, np.mean(training_cost)))


# the model can then be tested with the test set

feed = {tf_x: X_test_centered}

y_pred = sess.run(predictions['classes'], feed_dict=feed)

print('Test Accuracy: {}'.format(100*np.sum(y_pred == y_test) / y_test.shape[0]))
