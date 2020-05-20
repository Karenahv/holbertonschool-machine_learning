#!/usr/bin/env python3
""" mini-batch gradient descent"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid,
                     Y_valid, batch_size=32,
                     epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """trains with mini-batch method"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        train_op = tf.get_collection("train_op")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        feed_dict_train = {x: X_train, y: Y_train}
        feed_dict_valid = {x: X_valid, y: Y_valid}

        iterations = X_train.shape[0] / batch_size

        if isinstance(iterations, int):
            iterations = int(iterations)
        else:
            iterations = int(iterations) + 1

        for i in range(epochs + 1):
            cost_train = sess.run(loss,  feed_dict=feed_dict_train)
            acc_train = sess.run(accuracy,  feed_dict=feed_dict_train)
            cost_valid = sess.run(loss,  feed_dict=feed_dict_valid)
            acc_valid = sess.run(accuracy, feed_dict=feed_dict_valid)

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(acc_train))
            print("\tValidation Cost: {}".format(cost_valid))
            print("\tValidation Accuracy: {}".format(acc_valid))

            if i < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                for j in range(1, iterations + 1):
                    start = (j - 1) * batch_size
                    end = j * batch_size

                    if end > X_train.shape[0]:
                        end = X_train.shape[0]
                    feed_new = {x: X_shuffled[start:end],
                                y: Y_shuffled[start:end]}
                    sess.run(train_op, feed_dict=feed_new)
                    if j % 100 == 0:
                        cost_batch = sess.run(loss, feed_dict=feed_new)
                        acc_batch = sess.run(accuracy, feed_dict=feed_new)
                        print("\tStep {}:".format(j))
                        print("\t\tCost: {}".format(cost_batch))
                        print("\t\tAccuracy: {}".format(acc_batch))
        save_path = saver.save(sess, save_path)
    return save_path
