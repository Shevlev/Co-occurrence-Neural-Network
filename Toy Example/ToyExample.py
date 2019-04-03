'''
The final version of Toy Example:
this file include the train flow.
The user has to decide only what kind of network will be trained by setting the relevant string to TYPE_OF_NETWORK.
Also pay attention to TRAIN_DIR which defines where you trained network will be saved.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import numpy as np
import tensorflow as tf

import os

import create_data
import co_layer

'''TYPE_OF_NETWORK is string that set the type od network that you want to train.
TYPE_OF_NETWORK = 'FC'
TYPE_OF_NETWORK = 'Conv1x1x9'
TYPE_OF_NETWORK = 'Conv3x3x2'
TYPE_OF_NETWORK = 'Conv3x3x9'
TYPE_OF_NETWORK = 'CoNN'
'''
TYPE_OF_NETWORK = 'CoNN' # ''Conv3x3x2'
TRAIN_DIR = 'logsTest/'
BATCH_SIZE = 50
NUM_ITERATIONS = 2000


# def net(images, conv_size, type_layer):
def net(images, conv_size, conv_filters, conv_pool_avg,  type_layer):
    # reshape to the image size
    net_input = tf.reshape(images, [-1, create_data.SIZE_IMAGE, create_data.SIZE_IMAGE, 1])

    fc_filters = create_data.SIZE_IMAGE*create_data.SIZE_IMAGE

    conv_fc_filters = (10-conv_pool_avg+1)*(10-conv_pool_avg+1)*conv_filters
    conn_filters = 36
    conn_pool_avg = 5
    if type_layer == 'FC':
        # fuly-connected network
        flat_data = images
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([fc_filters, 36])
            flat1 = tf.matmul(flat_data, W_fc1)

        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([36, 2])
            b_fc2 = bias_variable([2])
            y_conv = tf.matmul(flat1, W_fc2) + b_fc2

    elif type_layer == 'Conv':
        # convolutional layer
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([conv_size, conv_size, 1, conv_filters])
            h_conv1 = tf.nn.relu(conv2d(net_input, W_conv1))

        input_dense = h_conv1
        # global average pooling
        p_avg = tf.nn.pool(input=input_dense,
                           window_shape=[conv_pool_avg, conv_pool_avg],
                           pooling_type="AVG",
                           padding="VALID")

        flat_data = tf.reshape(p_avg, [BATCH_SIZE, -1])

        # Map the 1024 features to 2 classes, one for each digit
        with tf.name_scope('fc_conv'):
            W_fc2 = weight_variable([conv_fc_filters, 2])
            b_fc2 = bias_variable([2])
            y_conv = tf.matmul(flat_data, W_fc2) + b_fc2

    elif type_layer == 'CoNN':
        # co-occurrence layer
        with tf.name_scope('conn'):
            CoL = co_layer.CoNN_layer(
                net_input,
                co_shape=[4, 4],
                co_initializer=None,
                w_shape=[10, 10, 1],
                w_initializer=None,
                name='col_')
            h_conv1 = tf.nn.relu(CoL)

        input_dense = h_conv1
        # global average pooling
        p_avg = tf.nn.pool(input=input_dense,
                           window_shape=[conn_pool_avg, conn_pool_avg],
                           pooling_type="AVG",
                           padding="VALID")

        flat_data = tf.reshape(p_avg, [BATCH_SIZE, -1])

        # Map the 1024 features to 2 classes, one for each digit
        with tf.name_scope('fc_conn'):
            W_fc2 = weight_variable([conn_filters, 2])
            b_fc2 = bias_variable([2])
            y_conv = tf.matmul(flat_data, W_fc2) + b_fc2

    else:
        assert (type_layer in xrange(3))

    y_pred_cls = tf.argmax(y_conv, axis=1, name='y_conv_max')

    return y_conv, y_pred_cls


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def train(type_layer, conv_size, conv_filters, conv_pool_avg, train_images, train_labels, test_images, test_labels, hparam):
    tf.reset_default_graph()

    # Create the model
    x = tf.placeholder(tf.float32, [None, create_data.SIZE_IMAGE * create_data.SIZE_IMAGE], name='image')

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None], name='type_hist')

    # Build the graph for the deep net
    y_conv, y_pred_cls = net(x, conv_size, conv_filters, conv_pool_avg, type_layer)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver()
    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # the summary of the graph and data
        writer = tf.summary.FileWriter(TRAIN_DIR + hparam)
        writer.add_graph(sess.graph)

        for i in range(NUM_ITERATIONS):
            randidx = np.random.randint(len(train_images), size=BATCH_SIZE)
            batch_xs = train_images[randidx]
            batch_ys = train_labels[randidx]

            train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

            if i % 100 == 0 or (i == NUM_ITERATIONS - 1):
                '''train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
                print('step %d, training accuracy %g' % (i, train_accuracy))'''
                [train_accuracy, s] = sess.run([train_step, merged_summary], feed_dict={x: batch_xs, y_: batch_ys})
                writer.add_summary(s, i)

            if i % 100 == 0 or (i == NUM_ITERATIONS - 1):
                # Make prediction for all images in test_x
                k = 0
                predicted_class = np.zeros(shape=len(test_images), dtype=np.int)
                while k < len(test_images):
                    j = min(k + BATCH_SIZE, len(test_images))
                    batch_xs = test_images[k:j, :]
                    batch_ys = test_labels[k:j]
                    predicted_class[k:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y_: batch_ys})
                    k = j

                correct = (test_labels == predicted_class)
                acc = correct.mean() * 100
                correct_numbers = correct.sum()
                print("Step " + str(i))
                print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_images)))

            saver.save(sess, os.path.join(TRAIN_DIR+hparam, "model.ckpt"))
        print("finish")


def main():
    # Clear the training directory
    if tf.gfile.Exists(TRAIN_DIR):
        tf.gfile.DeleteRecursively(TRAIN_DIR)
        tf.gfile.MakeDirs(TRAIN_DIR)

    # Import data
    print('Loading data')
    train_images, train_labels, test_images, test_labels = create_data.label_images()

    if TYPE_OF_NETWORK == 'FC':
        print('Test FC network')
        type_layer = 'FC'
        conv_size = 0
        conv_filters = 0
        conv_pool_avg = 0
        hparam = make_hparam_string(type_layer, conv_size, conv_filters, conv_pool_avg)
        train(type_layer, conv_size, conv_filters, conv_pool_avg, train_images, train_labels, test_images,
              test_labels, hparam)

    elif TYPE_OF_NETWORK == 'Conv1x1x9':
        print('Test Conv network with filter size 1x1x9')
        type_layer = 'Conv'
        conv_size = 1
        conv_filters = 9
        conv_pool_avg = 9  # 5- in order to get the same size of input to final fully-connected
        hparam = make_hparam_string(type_layer, conv_size, conv_filters, conv_pool_avg)
        train(type_layer, conv_size, conv_filters, conv_pool_avg, train_images, train_labels, test_images, test_labels,
              hparam)

    elif TYPE_OF_NETWORK == 'Conv3x3x2':
        print('Test Conv network with filter size 3x3x2')
        type_layer = 'Conv'
        conv_size = 3
        conv_filters = 2
        conv_pool_avg = 7
        hparam = make_hparam_string(type_layer, conv_size, conv_filters, conv_pool_avg)
        train(type_layer, conv_size, conv_filters, conv_pool_avg, train_images, train_labels, test_images, test_labels,
              hparam)

    elif TYPE_OF_NETWORK == 'Conv3x3x9':
        print('Test Conv network with filter size 3x3x9')
        type_layer = 'Conv'
        conv_size = 7
        conv_filters = 9
        conv_pool_avg = 7
        hparam = make_hparam_string(type_layer, conv_size, conv_filters, conv_pool_avg)
        train(type_layer, conv_size, conv_filters, conv_pool_avg, train_images, train_labels, test_images, test_labels,
              hparam)

    elif TYPE_OF_NETWORK == 'CoNN':
        print('Test CoNN')
        type_layer = 'CoNN'
        conv_size = 0
        conv_filters = 0
        conv_pool_avg = 0
        hparam = make_hparam_string(type_layer, conv_size, conv_filters, conv_pool_avg)
        train(type_layer, conv_size, conv_filters, conv_pool_avg, train_images, train_labels, test_images,
              test_labels, hparam)

    else:
        print('Error: The type of the network is UNKNOWN')


def make_hparam_string(type_layer, conv_size, conv_filters, conv_pool_avg):

    if type_layer == 'FC':
        str1 = "the_dense_layer"
    elif type_layer == 'Conv':
        str1 = "the_conv_layer_"
    elif type_layer == 'CoNN':
        str1 = "the_conn_layer"
    else:
        print("Error: the type of network is not valid")
        return "UNKNOWN"

    if conv_size == 0:
        str2 = ""
    else:
        str2 = "_with _size_" + str(conv_size)

    if conv_filters == 0:
        str3 = ""
    else:
        str3 = "_num_filters_" + str(conv_filters)

    if conv_pool_avg == 0:
        str4 = ""
    else:
        str4 = "_pool_" + str(conv_pool_avg)

    total_param = str1 + str2 + str3 + str4
    return "net_%s" % (total_param)

main()