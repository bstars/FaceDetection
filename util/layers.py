import sys
sys.path.append('..')


import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf


import util.config as cfg


def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=cfg.LEAKY_RELU_ALPHA)

def batch_norm(x, training:tf.placeholder):

    return tf.layers.batch_normalization(x, momentum=cfg.BATCH_NORM_MOMENTUM, epsilon=cfg.BATCH_NORM_EPS,
                                         trainable=True, training=training, fused=False)

def conv_norm_relu(x, training, filters, kernel_size, strides):
    net = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME', use_bias=True)
    net = batch_norm(net, training)
    net = leaky_relu(net)
    return net

def yolo_conv_block(x, training, filters):
    """
    route with same channels as input
    net double the number of channels of input
    """
    net = conv_norm_relu(x,   training=training, filters=filters * 1, kernel_size=1, strides=1)
    net = conv_norm_relu(net, training=training, filters=filters * 2, kernel_size=3, strides=1)
    net = conv_norm_relu(net, training=training, filters=filters * 1, kernel_size=1, strides=1)
    net = conv_norm_relu(net, training=training, filters=filters * 2, kernel_size=3, strides=1)
    net = conv_norm_relu(net, training=training, filters=filters * 1, kernel_size=1, strides=1)

    route = net

    net = conv_norm_relu(net, training=training, filters=filters * 2, kernel_size=3, strides=1)
    return route, net


def yolo_detection_block(x, n_anchors):
    net = tf.layers.conv2d(x, filters=n_anchors * 5, kernel_size=1, strides=1, padding='SAME')
    cell_size = net.get_shape().as_list()[1]
    net = tf.reshape(net, shape=[-1, cell_size, cell_size, n_anchors, 5])

    return net

def unsample_layer(x, output_shape):
    new_height = output_shape[1]
    new_width = output_shape[2]
    net = tf.image.resize_nearest_neighbor(x, (new_height, new_width))
    return net



