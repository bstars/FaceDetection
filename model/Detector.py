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


from model.FaceDetection import FaceDetectionNet
import util.config as cfg


class Detector(object):
    def __init__(self):
        self.ckpt_path = cfg.CKPT_PATH
        self.anchors = cfg.ANCHORS
        self.img_size = cfg.IMG_SIZE
        self.SMALL_SIZE = cfg.SMALL_SIZE
        self.MID_SIZE = cfg.MID_SIZE
        self.LARGE_SIZE = cfg.LARGE_SIZE
        self.CONFIDENCE_THRESHOLD = cfg.CONFIDENCE_THRESHOLD
        self.IOU_THRESHOLD = cfg.IOU_THRESHOLD


        self.net = FaceDetectionNet()
        self.sess = tf.Session(graph=self.net.graph)

        with self.net.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, self.ckpt_path)


    def predict(self, img:np.ndarray):

        image = np.expand_dims(img, axis=0)
        feed_dict = {
            self.net.images : image,
            self.net.training : False
        }
        with self.net.graph.as_default():
            label_small, label_mid, label_large = self.sess.run(
                [self.net.detect1, self.net.detect2, self.net.detect3],
                feed_dict=feed_dict
            )
            # label_flatten = self.sess.run(self.net.detect_flatten,feed_dict=feed_dict)
            # label_small = np.reshape(label_flatten[:, :14*14*3*5], newshape=[1,14,14,3,5])
            # label_mid = np.reshape(label_flatten[:, 14*14*3*5:14*14*3*5+28*28*3*5], newshape=[1,28, 28, 3, 5])
            # label_large = np.reshape(label_flatten[:, 14*14*3*5+28*28* 3 * 5:], newshape=[1,56, 56, 3, 5])

        label_small = self.interpret_predict(label_small[0], self.anchors[6:9])
        label_mid = self.interpret_predict(label_mid[0], self.anchors[3:6])
        label_large = self.interpret_predict(label_large[0],self.anchors[0:3])

        boxes = self.get_boxes(label_small, label_mid, label_large)
        if len(boxes) != 0:
            boxes = self.nonmax_suppression(boxes)
        return boxes

    def nonmax_suppression(self, boxes):
        # TODO: Optimize to reduce the complexity
        confidences = boxes[:,0]
        argsort = np.array(np.argsort(confidences))[::-1]
        boxes = boxes[argsort]
        for i in range(len(boxes)):
            if boxes[i,0] == 0:
                continue
            for j in range(i + 1, len(boxes)):
                if self.iou(boxes[i,1:], boxes[j,1:]) > self.IOU_THRESHOLD:
                    boxes[j,0] = 0.0

        confidences = boxes[:,0]
        idx = np.array(confidences > 0.0, dtype=bool)
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
        xcenter1, ycenter1, w1, h1 = box1
        xcenter2, ycenter2, w2, h2 = box2

        if xcenter1 + w1 / 2 < xcenter2 - w2 / 2  or xcenter2 + w2 / 2 < xcenter1 - w1 / 2:
            return 0.

        if ycenter1 + h1 / 2 < ycenter2 - h2 / 2  or ycenter2 + h2 / 2 < ycenter1 - h1 / 2:
            return 0.

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

def plot_img_with_boxes_cv(img, boxes):
    """
        :param img:
        :param boxes: a list of boxes [xcenter, ycenter, w, h]
        :return:
        """

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float) / 255

    for box in boxes:
        confidence, xcenter, ycenter, width, height = box
        x1 = int(xcenter - width / 2)
        y1 = int(ycenter - height / 2)
        x2 = int(xcenter + width / 2)
        y2 = int(ycenter + height / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,1), 2)
        cv2.putText(image, " %.2f" % (confidence), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,1), 2, cv2.LINE_AA)
    return (image * 255).astype(np.uint8)


def generate_video(images):
    out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    for img in images:
        out.write(img)
        # out.write(cv2.Canny(img,1,100))
    out.release()


if __name__ == "__main__":
    # fname = '../test/56.jpeg'
    # 45 71 74 76

    fname = os.path.join(cfg.TEST_DATA_PATH, str(1) + '.jpg')
    # img = cv2.imread(fname, cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = plt.imread(fname)
    img = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE))

    detector = Detector()
    t = time.time()
    boxes = detector.predict(img)
    elapsed = time.time() - t
    print(elapsed)

    plot_img_with_boxes(img, boxes)





