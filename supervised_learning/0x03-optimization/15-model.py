#!/usr/bin/env python3
"""builds, trains, and save a neural model """

import tensorflow as tf
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer
create_Adam_op = __import__('10-Adam').create_Adam_op
learning_rate_decay = __import__('12-learning_rate_decay').learning_rate_decay


def forward_prop(x, layer_sizes=[], activations=[]):
    """ creates the forward propagation graph for the neural network"""
    new_layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for layer in range(1, len(layer_sizes)):
        new_layer = create_batch_norm_layer(new_layer,
                                            layer_sizes[layer],
                                            activations[layer])
    return new_layer


def calculate_accuracy(y, y_pred):
    """calculates accuracy of the prediction"""
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999,
          epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """builds, trains, and save a neural model """

    # Build model
    x = (tf.placeholder(tf.float32,
         shape=(None, Data_train[0].shape[1]), name='x'))
    y = (tf.placeholder(tf.float32,
         shape=(None, Data_train[1].shape[1]), name='y'))
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    # Train Adam and learning decay
    global_step = tf.Variable(0, trainable=False, name='global_step')
    tf.add_to_collection('global_step', global_step)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        feed_dict_train = {x: Data_train[0], y: Data_train[1]}
        feed_dict_valid = {x: Data_valid[0], y: Data_valid[1]}
        iterations = Data_train[0].shape[0] / batch_size
        if isinstance(iterations, int):
            iterations = int(iterations)
        else:
            iterations = int(iterations) + 1
        for iter in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict=feed_dict_train)
            acc_train = sess.run(accuracy, feed_dict=feed_dict_train)
            cost_valid = sess.run(loss, feed_dict=feed_dict_valid)
            acc_valid = sess.run(accuracy, feed_dict=feed_dict_valid)
            print("After {} epochs:".format(iter))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(acc_train))
            print("\tValidation Cost: {}".format(cost_valid))
            print("\tValidation Accuracy: {}".format(acc_valid))
            if iter < epochs:
                X_shuffled, Y_shuffled = (shuffle_data(Data_train[0],
                                          Data_train[1]))
                sess.run(global_step.assign(iter))
                sess.run(alpha)
                for step in range(iterations):
                    start = step*batch_size
                    end = (step + 1)*batch_size
                    if end > Data_train[0].shape[0]:
                        end = Data_train[0].shape[0]
                    feed_dict_mini = {x: X_shuffled[start: end],
                                      y: Y_shuffled[start: end]}
                    sess.run(train_op, feed_dict_mini)
                    if step != 0 and (step + 1) % 100 == 0:
                        print("\tStep {}:".format(step + 1))
                        cost_mini = sess.run(loss, feed_dict_mini)
                        print("\t\tCost: {}".format(cost_mini))
                        acc_mini = sess.run(accuracy, feed_dict_mini)
                        print("\t\tAccuracy: {}".format(acc_mini))
        save_path = saver.save(sess, save_path)
    return save_path
