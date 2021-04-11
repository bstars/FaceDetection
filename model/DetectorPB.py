import sys
sys.path.append('..')

import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import time
import os

import util.config as cfg

class DetectorPB(object):
    def __init__(self):
        self.ckpt_path = cfg.CKPT_PATH
        self.anchors = cfg.ANCHORS
        self.img_size = cfg.IMG_SIZE
        self.SMALL_SIZE = cfg.SMALL_SIZE
        self.MID_SIZE = cfg.MID_SIZE
        self.LARGE_SIZE = cfg.LARGE_SIZE
        self.CONFIDENCE_THRESHOLD = cfg.CONFIDENCE_THRESHOLD
        self.IOU_THRESHOLD = cfg.IOU_THRESHOLD


        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            f = tf.gfile.GFile('../frozen/inference_graph.pb', 'rb')
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            self.images, self.training, self.detect1, self.detect2, self.detect3 = tf.import_graph_def(graph_def,
                                                                              return_elements=[cfg.INPUT_NAME,
                                                                                               cfg.TRAINING_NAME,
                                                                                               cfg.DETECT1_NAME,
                                                                                               cfg.DETECT2_NAME,
                                                                                               cfg.DETECT3_NAME])

    def predict(self, img:np.ndarray):
        image = np.expand_dims(img, axis=0)
        feed_dict = {
            self.images : image,
            self.training : False
        }
        with self.graph.as_default():
            label_small, label_mid, label_large = self.sess.run(
                [self.detect1, self.detect2, self.detect3],
                feed_dict=feed_dict
            )

        label_small = self.interpret_predict(label_small[0], self.anchors[6:9])
        label_mid = self.interpret_predict(label_mid[0], self.anchors[3:6])
        label_large = self.interpret_predict(label_large[0], self.anchors[0:3])

        boxes = self.get_boxes(label_small, label_mid, label_large)
        if len(boxes) != 0:
            boxes = self.nonmax_suppression(boxes)
        return boxes

    def nonmax_suppression(self, boxes):
        # TODO: Optimization to reduce the complexity
        confidences = boxes[:,0]
        argsort = np.array(np.argsort(confidences))[::-1]
        boxes = boxes[argsort]
        for i in range(len(boxes)):
            if boxes[i,0] == 0:
                continue
            for j in range(i + 1, len(boxes)):
                if self.iou(boxes[i,1:], boxes[j,1:]) > self.IOU_THRESHOLD:
                    boxes[j,0] = 0.0
                    # np.delete(boxes, j)
        confidences = boxes[:,0]
        idx = np.array(confidences > 0.0, dtype=bool)
        print('idx.shape', idx.shape)
        return boxes[idx]

    def get_boxes(self, label_small, label_mid, label_large):
        labels = [label_small, label_mid, label_large]
        boxes = []

        for label in labels:
            confidence = label[:, :, :, 0]
            Is, Js, Ks = np.where(confidence >= self.CONFIDENCE_THRESHOLD)

            n_boxes = len(Is)
            for i in range(n_boxes):
                box = label[Is[i], Js[i], Ks[i], :]
                boxes.append(box)
        return np.array(boxes)

    def interpret_predict(self, label, anchors):
        """
        :param label: [grid_size, grid_size, 3, 5]
        :param anchors: [(),(),()]
        :return:
        """
        grid_size = label.shape[0]
        px_per_cell = float(self.img_size) / grid_size


        confidence = label[...,0]
        confidence = np.expand_dims(confidence, axis=-1)
        xy = label[...,1:3]
        wh = label[...,3:5]
        # print(confidence)

        # print(confidence.shape, xy.shape, wh.shape)

        offset = self.get_offset(grid_size, n_anchors=3)
        xy = (xy + offset) * px_per_cell

        wh = wh * anchors
        return np.concatenate([confidence, xy, wh], axis=-1)

    def get_offset(self, grid_size, n_anchors=3):
        x = np.arange(0, grid_size)
        y = np.arange(0, grid_size)
        xx, yy = np.meshgrid(x, y)
        offset = np.stack([xx, yy], axis=-1)
        offset = np.expand_dims(offset, axis=2)
        offset = np.tile(offset, [1,1,n_anchors,1])
        return offset

    def iou(self, box1, box2):
        """

        :param box1: (xcenter, ycenter, w, h)
        :param box2: (xcenter, ycenter, w, h)
        :return:
        """

        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

def plot_img_with_boxes(img, boxes):
    """
    :param img:
    :param boxes: a list of boxes [xcenter, ycenter, w, h]
    :return:
    """
    fig, ax = plt.subplots(1)
    ax.imshow(img.astype(int))
    for box in boxes:
        confidence, xcenter, ycenter, width, height = box
        x1, y1 = xcenter - width / 2, ycenter - height / 2
        x2, y2 = xcenter + width / 2, ycenter + height / 2

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        x1 = xcenter - width / 2
        y1 = ycenter - height / 2
        ax.text(x1, y1, " %.2f" % (confidence), bbox=dict(facecolor='red', alpha=0.9))

    plt.show()

if __name__ == "__main__":
    # fname = '../test/56.jpeg'
    fname = os.path.join(cfg.TEST_DATA_PATH, '45.jpeg')
    img = cv2.imread(fname)
    img = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    detector = DetectorPB()


    t = time.time()
    boxes = detector.predict(img)
    elapsed = time.time() - t

    print(elapsed)

    plot_img_with_boxes(img, boxes)
