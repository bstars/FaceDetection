import sys
sys.path.append('..')

import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf


from model.FaceDetection import FaceDetectionNet
from util.fddb import FDDBDataSet
import util.config as cfg


class Solver(object):
    def __init__(self, net:FaceDetectionNet, fddb:FDDBDataSet):
        self.net = net
        self.fddb = fddb
        self.lrs = [1e-5] * 20

        self.batch_size = cfg.BATCH_SIZE
        self.vgg_path = cfg.VGG_PATH

        with self.net.graph.as_default():
            self.learning_rate_holder = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_holder)

            self.train_op = self.optimizer.minimize(self.net.loss_reg, var_list=self.net.get_trainable_vars())
            self.train_op = tf.group([self.train_op, update_ops])

    def fit(self, max_epoch):
        graph = self.net.graph
        sess = tf.Session(graph=graph)
        with graph.as_default():
            sess.run(tf.global_variables_initializer())
            self.net.load_vgg_weights(self.vgg_path, sess)

        iter = 0
        while self.fddb.epoch < max_epoch:
            iter += 1
            lr = self.lrs[self.fddb.epoch]

            imgs, labels_small, labels_mid, labels_large = self.fddb.get(self.batch_size)
            feed_dict = {
                self.net.images : imgs,
                self.net.labels_small : labels_small,
                self.net.labels_medium : labels_mid,
                self.net.labels_large : labels_large,
                self.net.training : True,
                self.learning_rate_holder : lr
            }

            _, l, l_reg = sess.run([self.train_op, self.net.loss, self.net.loss_reg],
                                   feed_dict=feed_dict)
            print("Epoch %f, \t iter %d, \t loss %.6f \t loss %.6f" %
                  (self.fddb.epoch + self.fddb.cursor / self.fddb.m, iter, l, l_reg))

if __name__ == "__main__":
    fddb = FDDBDataSet()
    net = FaceDetectionNet()
    solver = Solver(net, fddb)
    solver.fit(20)