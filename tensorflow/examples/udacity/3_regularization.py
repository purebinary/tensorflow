# !/usr/bin/python
# -*- coding:utf-8 -*-

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


num_nodes = 1024
batch_size = 128
beta = 5e-4

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights_1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_nodes], stddev=0.051))
    biases_1 = tf.Variable(tf.zeros([num_nodes]))

    weights_2 = tf.Variable(
        tf.truncated_normal([num_nodes, 1024], stddev=0.042))
    biases_2 = tf.Variable(tf.zeros([1024]))

    weights_3 = tf.Variable(
        tf.truncated_normal([1024, 128], stddev=0.055))
    biases_3 = tf.Variable(tf.zeros([128]))

    weights_4 = tf.Variable(
        tf.truncated_normal([128, num_labels], stddev=0.075))
    biases_4 = tf.Variable(tf.zeros([num_labels]))

    hidden1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1), 0.75)
    hidden2 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden1, weights_2) + biases_2), 0.65)
    hidden3 = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden2, weights_3) + biases_3), 0.5)
    # hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
    # hidden2 = tf.nn.relu(tf.matmul(hidden1, weights_2) + biases_2)

    valid_hidden1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1)
    valid_hidden2 = tf.nn.relu(tf.matmul(valid_hidden1, weights_2) + biases_2)
    valid_hidden3 = tf.nn.relu(tf.matmul(valid_hidden2, weights_3) + biases_3)

    test_hidden1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
    test_hidden2 = tf.nn.relu(tf.matmul(test_hidden1, weights_2) + biases_2)
    test_hidden3 = tf.nn.relu(tf.matmul(test_hidden2, weights_3) + biases_3)

    logits = tf.matmul(hidden3, weights_4) + biases_4
    #l2_loss =  tf.nn.l2_loss(weights_4)
    l2_loss = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + beta * l2_loss

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.5, global_step, 5000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(valid_hidden3, weights_4) + biases_4)
    test_prediction = tf.nn.softmax(
        tf.matmul(test_hidden3, weights_4) + biases_4)

num_steps = 50001
#l = 10
#step = 0
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
  #while l > 0.4:
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    #step += 1
    if (step % 5000 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))