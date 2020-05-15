#!/usr/bin/env python3
"""evaluate the output of a NN """

import tensorflow as tf


def evaluate(X, Y, save_path):
    """evaluate the output of a NN"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        feed_dict = {x: X, y: Y}
        final_p = sess.run(y_pred, feed_dict)
        acc_final = sess.run(accuracy, feed_dict)
        lss_final = sess.run(loss, feed_dict)

        return final_p, acc_final, lss_final
