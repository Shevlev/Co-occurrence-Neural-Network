
'''
Co-occurrence layer with one 2d matrix of co-occurrence filter and classic convolutional filter.
The number of output channels can be chosen by the user and implemented only in convolutional part.
the conv filter : [filter_size[0], filter_size[1], filter_size[2]]
the cof size: [number_bins, number_bins]
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf


def conn_variable_initializer(initializer, shape, name):
    """initialization of the filters"""
    if (initializer is None):
        initial = tf.truncated_normal(shape, stddev=0.1)
    else:
        initial = initializer(shape)
    return tf.Variable(initial, name=name)


# -----CoNN sample-------------------------------------
def CoNN_layer(layer_input, co_shape, co_initializer, w_shape, w_initializer, strides=1, name=""):
    """
    :param layer_input: input to layer
    :param co_shape: shape of 2D co-occurrence matrix (type: int) for example [5, 5]
    :param co_initializer:
             tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32)
             tf.constant_initializer(0.05)
    :param w_shape: shape of 3D spatial filter (type int) for example [3, 3, 3]
    :param w_initializer:
            tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32)
            tf.constant_initializer(0.05)
    :param strides: the strides for CoL implementation like in the convutional layer
    :param name: for extracting the layer components in Tensorflow
           (for debugging and visualization), for example "col_"
    """

    co_filter = conn_variable_initializer(co_initializer, co_shape, name+'co_filter')
    # cof_filter = tf.constant(value=1.0, dtype=tf.float32, shape=cof_shape)

    # w_spatial = conn_variable_initializer(w_initializer, w_shape, name+'w_filter')
    w_spatial = tf.constant(value=1.0, dtype=tf.float32, shape=w_shape)

    # the size parameters
    num_bins = co_shape[0]
    x_shape = tf.shape(layer_input)

    # quantization step of the input for index decision
    input_index = quantization_input_as_bins(layer_input, num_bins)

    vec_input_index = tf.reshape(input_index, [-1, 1])

    # the variable with final result
    sp0, sp1, sp2, sp3 = layer_input.get_shape().as_list()
    sp1 = int(sp1 / strides)

    total_conv = tf.zeros([x_shape[0], sp1, sp1, sp3])

    temp_strides = tf.zeros([strides])
    # while loop of bins in co-occurrence matrix
    condition_per_bins = lambda i, *_: i < num_bins
    _, total_conv, _, _, _, _, _, _ = tf.while_loop(condition_per_bins, body_per_bins,
                                                 [0, total_conv, w_spatial, co_filter, layer_input, input_index,
                                                  vec_input_index, temp_strides])

    CoL = total_conv

    return CoL


def body_per_bins(i, total_conv, w_spatial, co_filter, x, input_index, vec_input_index, strides):
    """
    the function is a body of the loop that run on the size of a co-occurrence matrix
    :param i: iteration number
    :param total_conv: final variable that contains the result
    :param w_spatial: the spatial filter
    :param co_matrix: the co-occurrence matrix
    :param x: the input of the layer
    :param input_index: the quantized input for the bins of co-occurrence matrix
    :param vec_input_index: reshaped input_index
    :param strides: optionally can be implemented stride
    :return:
    """

    # shape of input data
    x_shape = tf.shape(x)
    w_shape = tf.shape(w_spatial)
    # fx, fy, fin, fout = w_spatial.get_shape().as_list()
    # take only the line with specific bin
    bin_row = co_filter[i, :]

    cof_matrix_bin = tf.reshape(tf.gather_nd(bin_row, vec_input_index), x_shape)

    temp_input = tf.multiply(x, cof_matrix_bin)
    temp_input = tf.expand_dims(temp_input, axis=4)
    w_3d = tf.reshape(w_spatial, [w_shape[0], w_shape[1], w_shape[2], 1, 1])

    strid = strides.get_shape().as_list()
    temp_mask = tf.cast(tf.equal(input_index, i), tf.float32)
    temp_mask = tf.expand_dims(temp_mask, axis=4)
    filter_ones = tf.ones([1, 1, 1, 1, 1])
    temp_mask = tf.nn.conv3d(temp_mask, filter_ones, strides=[1, strid[0], strid[0], 1, 1], padding="SAME")
    temp_result = tf.nn.conv3d(temp_input, w_3d, strides=[1, strid[0], strid[0], 1, 1], padding="SAME")
    # temp_result = tf.reduce_sum(temp_result, axis=4)
    temp_result = tf.multiply(temp_result[:, :, :, :, 0], temp_mask[:, :, :, :, 0])
    total_conv = total_conv+temp_result

    return i + 1, total_conv, w_spatial, co_filter, x, input_index, vec_input_index, strides


# the uniform quantization function
def quantization_input_as_bins(x, k_center):
    """
    quantization the input x to the k-bins structure
    :param x: the input data
    :param k_center: the number of bins
    :return: quantized input
    """

    # normalization input x to the k-bins vector
    x = tf.cast(x, tf.float32)
    x_shape = tf.shape(x)
    x_ones = tf.ones_like(x)  # create tensor the same shape as x

    x_min_per_batch = tf.reduce_min(x, axis=0)  # min per batch
    x_min_per_image = tf.reduce_min(tf.reduce_min(x_min_per_batch, axis=0), axis=0)
    x_min_global = tf.reduce_min(x_min_per_image)
    x_min_mat = tf.multiply(x_min_global, x_ones)
    x_idx_0 = tf.subtract(x, x_min_mat)
    assert_zeros1 = tf.assert_non_negative(x_idx_0)
    with tf.control_dependencies([assert_zeros1]):
        x_idx_0 = tf.identity(x_idx_0, name='x_idx_0')

    x_max_per_batch = tf.reduce_max(x_idx_0, axis=0)
    x_max_per_image = tf.reduce_max(tf.reduce_max(x_max_per_batch, axis=0), axis=0)
    x_max_global = tf.reduce_max(x_max_per_image, axis=0)
    x_max_mat = tf.multiply(x_max_global, x_ones)
    x_idx_01 = tf.div(x_idx_0, x_max_mat)  # TODO
    assert_zeros2 = tf.assert_non_negative(x_idx_01)
    with tf.control_dependencies([assert_zeros2]):
        x_idx_01 = tf.identity(x_idx_01, name='x_idx_01')

    # [0..1]*(256-1)->[0..255]+1->[1..256] and then floor getting the interger not greater than x
    k = tf.cast((k_center - 1), tf.float32)
    x_idx = tf.multiply(x_idx_01, k)
    x_idx = tf.round(x_idx)

    # x_idx = tf.floor(tf.add(x_idx,1))
    # x_idx = tf.floor(x_idx)  # the range [0..255]because we have z index in the python list
    x_idx = tf.cast(x_idx, tf.int32)
    # indices have to be non negative
    assert_zeros = tf.assert_non_negative(x_idx)
    with tf.control_dependencies([assert_zeros]):
        x_idx = tf.identity(x_idx, name='x_idx')

    return x_idx

