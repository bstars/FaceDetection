import sys
sys.path.append('..')

import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf
import numpy as np

import util.config as cfg
import util.layers as layers

class FaceDetectionNet(object):
    def __init__(self):

        self.anchors = cfg.ANCHORS
        self.img_size = cfg.IMG_SIZE
        self.object_scale= cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.coord_scale= cfg.COORD_SCALE
        self.reg = cfg.REG

        self.SCALE_SMALL_OBJS = cfg.SCALE_SMALL_OBJS
        self.SCALE_MID_OBJS = cfg.SCALE_MID_OBJS
        self.SCALE_LARGE_OBJS = cfg.SCALE_LARGE_OBJS

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.images = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size, 3], name='images')
            self.training = tf.placeholder(dtype=tf.bool, shape=(), name='training')
            # self.training = False

            self.labels_small = tf.placeholder(dtype=tf.float32, shape=[None, 14, 14, 3, 5], name='labels_small')
            self.labels_medium = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3, 5], name='labels_medium')
            self.labels_large = tf.placeholder(dtype=tf.float32, shape=[None, 56, 56, 3, 5], name='labels_large')

            with tf.variable_scope('vgg_conv_layers'):
                # (?, 56, 56, 256) (?, 28, 28, 512) (?, 14, 14, 512)
                self.route1, self.route2, self.route3 = self.build_vgg_layers(self.images)

            with tf.variable_scope('yolo_v3'):
                # (?, 14, 14, 3, 5)(?, 28, 28, 3, 5) (?, 56, 56, 3, 5)
                self.detect1, self.detect2, self.detect3 = self.build_yolo_layers(self.route1, self.route2, self.route3)
                detect1_flatten = tf.layers.flatten(self.detect1)
                detect2_flatten = tf.layers.flatten(self.detect2)
                detect3_flatten = tf.layers.flatten(self.detect3)

            self.detect_flatten = tf.concat([detect1_flatten, detect2_flatten,detect3_flatten], axis=-1, name='detect_flatten')


            with tf.variable_scope('loss'):
                self.loss_small = self.build_loss(self.labels_small, self.detect1, anchor_type='large') # small labels, large objs
                self.loss_mid = self.build_loss(self.labels_medium, self.detect2, anchor_type='medium') # mid labels, mid objs
                self.loss_large = self.build_loss(self.labels_large, self.detect3, anchor_type='small') # large labes, small objs

                self.loss = self.loss_small * self.SCALE_LARGE_OBJS \
                            + self.loss_mid * self.SCALE_MID_OBJS\
                            + self.loss_large * self.SCALE_SMALL_OBJS

                self.loss_reg = self.loss
                for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    if 'kernel' in i.name:
                        self.loss_reg += tf.nn.l2_loss(i) * self.reg

    def load_vgg_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i >= len(self.vgg_parameters):
                return
            print(i, k, np.shape(weights[k]))
            sess.run(self.vgg_parameters[i].assign(weights[k]))

    def build_vgg_layers(self, input):

        # the following codes are from https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
        # with a little bit modification to make it compatible with Movidius NCS

        self.vgg_parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = tf.subtract(input, mean)

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')
        # print('pool1',pool1.shape)

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')
        # print('pool2', pool2.shape)

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # pool3
        pool3 = tf.nn.max_pool(conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')
        # print('pool3', pool3.shape)

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # pool4
        pool4 = tf.nn.max_pool(conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')
        # print('pool4', pool4.shape)

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv5_1 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv5_2 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.add(conv, biases)
            conv5_3 = tf.nn.relu(out, name=scope)
            self.vgg_parameters += [kernel, biases]

        # pool5
        pool5 = tf.nn.max_pool(conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool5')
        # print('pool5', pool5.shape)

        # (?, 56, 56, 256) (?, 28, 28, 512) (?, 14, 14, 512)
        return pool3, pool4, pool5

    def build_yolo_layers(self, route1, route2, route3):
        route, net = layers.yolo_conv_block(route3, training=self.training, filters=512)

        # (?, 14, 14, 3, 5)
        detect1 = layers.yolo_detection_block(net, n_anchors=3)

        # (?, 14, 14, 256)
        net = layers.conv_norm_relu(route, training=self.training, filters=256, kernel_size=1, strides=1)
        unsample_shape = route2.get_shape().as_list()

        # (?, 28, 28, 256)
        net = layers.unsample_layer(net, unsample_shape)

        # (?, 28, 28, 768)
        net = tf.concat([net, route2], axis=-1)

        # (?, 28, 28, 256)(?, 28, 28, 512)
        route, net = layers.yolo_conv_block(net, training=self.training, filters=256)

        detect2 = layers.yolo_detection_block(net, n_anchors=3)

        # (?, 28, 28, 128)
        net = layers.conv_norm_relu(route, training=self.training, filters=128, kernel_size=1, strides=1)
        unsample_shape = route1.get_shape().as_list()

        # (?, 56, 56, 128)
        net = layers.unsample_layer(net, unsample_shape)

        # (?, 56, 56, 384)
        net = tf.concat([net, route1], axis=-1)

        # (?, 56, 56, 128) (?, 56, 56, 256)
        route, net = layers.yolo_conv_block(net, training=self.training, filters=128)

        # (?, 56, 56, 3, 5)
        detect3 = layers.yolo_detection_block(net, 3)

        return detect1, detect2, detect3

    def build_loss(self, labels, detects, anchor_type:str):
        """

        :param label: of shape [-1, cell_size, cell_size, 3, 5]
        :param detect:
        :return:
        """
        n_anchors = 3
        if anchor_type == 'large':
            anchors = self.anchors[6:9]
        elif anchor_type == 'medium':
            anchors = self.anchors[3:6]
        elif anchor_type == 'small':
            anchors = self.anchors[0:3]
        else:
            raise Exception()

        cell_size = labels.get_shape().as_list()[1]
        px_per_cell = float(self.img_size) / cell_size

        offset = self.get_offset(cell_size, n_anchors)
        anchors = tf.to_float(anchors)
        anchors = tf.expand_dims(anchors, axis=0)
        anchors = tf.expand_dims(anchors, axis=0)
        anchors = tf.tile(anchors, [cell_size, cell_size,1,1])


        true_confidences, true_xy, true_wh = tf.split(labels, [1,2,2], axis=-1)
        confidences, xy, wh = tf.split(detects, [1,2,2], axis=-1)

        # (cell size, cell size, n_anchors, 2)
        true_xy_tran = tf.subtract(tf.divide(true_xy, px_per_cell), offset)
        true_wh_tran = tf.divide(true_wh, anchors)

        # object mask   (?, cell size, cell size, 3, 1)
        object_mask = tf.to_float(true_confidences >= 1)
        noobject_mask = tf.subtract(tf.ones_like(object_mask), object_mask)


        # convert predict box to absolute pixel, used to calculate iou
        box_predict_tran = tf.stack([
            tf.multiply(tf.add(detects[...,1], offset[...,0]), px_per_cell),
            tf.multiply(tf.add(detects[...,2], offset[...,1]), px_per_cell),
            tf.multiply(detects[...,3], anchors[...,0]),
            tf.multiply(detects[...,4], anchors[..., 1])
        ], axis=-1)

        ious = self.calculate_iou(box_predict_tran, labels[...,1:])
        # (?, cell size, cell size, 3, 1)
        ious = tf.expand_dims(ious, axis=-1)

        # object loss   (?, cell size, cell size, 3, 1)
        object_delta = tf.multiply(tf.subtract(confidences, ious), object_mask)
        object_loss = tf.reduce_mean(
            tf.reduce_sum(object_delta**2, axis=[1,2,3,4])
        ) * self.object_scale

        # noobject loss (?, 14, 14, 3, 1)
        noobject_delta = tf.multiply(confidences, noobject_mask)
        noobject_loss = tf.reduce_mean(
            tf.reduce_sum(noobject_delta**2, axis=[1,2,3,4])
        ) * self.noobject_scale

        # xy loss   (?, cell size, cell size, 3, 1)
        xy_loss = tf.reduce_sum(
            tf.square(tf.subtract(true_xy_tran, xy)),
            axis=-1, keepdims=True)
        xy_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(xy_loss, object_mask),
                axis=[1,2,3,4])
        ) * self.coord_scale

        # wh loss
        wh_loss = tf.reduce_sum(
            tf.square(tf.subtract(true_wh_tran, wh)),
            axis=-1, keepdims=True)
        wh_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(wh_loss, object_mask),
                axis=[1,2,3,4])
        ) * self.coord_scale

        return object_loss + noobject_loss + xy_loss + wh_loss
        # return xy_loss + wh_loss

    def get_offset(self, cell_size:int, num_anchors:int):
        x = tf.range(cell_size, dtype=tf.float32)
        y = tf.range(cell_size, dtype=tf.float32)
        xx, yy = tf.meshgrid(x, y)
        offset = tf.stack([xx, yy], axis=-1)
        offset = tf.expand_dims(offset, axis=2)
        offset = tf.tile(offset, [1,1,num_anchors,1])
        return offset

    def get_trainable_vars(self):
        vars = []
        with self.graph.as_default():
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                if 'yolo' in i.name:
                    vars.append(i)
        return vars

    def calculate_iou(self, box1, box2):
        """
        :param box1: [batch size, cell size, cell size, box per cell, 4]
        :param box2: [batch size, cell size, cell size, box per cell, 4]
        :return:
        """

        # upper-left and lower-right point
        box1_tran = tf.stack([
            box1[:,:,:,:,0] - box1[:,:,:,:,2] / 2.,
            box1[:,:,:,:,1] - box1[:,:,:,:,3] / 2.,
            box1[:,:,:,:,0] + box1[:,:,:,:,2] / 2.,
            box1[:,:,:,:,1] + box1[:,:,:,:,3] / 2.,
        ], axis=-1)

        box2_tran = tf.stack([
            box2[:,:,:,:,0] - box2[:,:,:,:,2] / 2.,
            box2[:,:,:,:,1] - box2[:,:,:,:,3] / 2.,
            box2[:,:,:,:,0] + box2[:,:,:,:,2] / 2.,
            box2[:,:,:,:,1] + box2[:,:,:,:,3] / 2.,
        ], axis=-1)

        lu = tf.maximum(box1_tran[..., :2], box2_tran[..., :2])
        rd = tf.minimum(box1_tran[..., 2:], box2_tran[..., 2:])

        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = box1[..., 2] * box1[..., 3]
        square2 = box2[..., 2] * box2[..., 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

if __name__ == '__main__':
    model = FaceDetectionNet()
