import sys
sys.path.append('..')

from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import cv2


import util.config as cfg



class FDDBDataSet(object):
    def __init__(self):
        self.SMALL_SIZE = cfg.SMALL_SIZE
        self.MID_SIZE = cfg.MID_SIZE
        self.LARGE_SIZE = cfg.LARGE_SIZE
        self.anchors = cfg.ANCHORS
        self.IMG_SIZE = cfg.IMG_SIZE
        self.AUGMENTS = ['IDENTITY', 'BLUR', 'COLOR', 'AVERAGE', 'GAUSSIAN', "MEDIAN"]
        # self.AUGMENTS = ['IDENTITY', 'FLIP','BLUR', 'AVERAGE', 'GAUSSIAN', "MEDIAN"]
        self.epoch = 0
        self.cursor = 0


        self.path = cfg.FDDB_PATH
        self.folds = ['01','02','03','04','05','06','07','08','09','10']
        self.idxes = self.build_indexes()
        self.m = len(self.idxes)

    def build_indexes(self):
        idxes = []
        for fold in self.folds:
            idx_filename = os.path.join(self.path, 'FDDB-folds', 'FDDB-fold-' + fold + '.txt')
            idx_file = open(idx_filename)

            for line in idx_file:
                idxes.append(line.strip())
        return idxes

    def get_single_data(self, idx):

        image_path = os.path.join(self.path, 'originalPics', idx + '.jpg')
        label_path = os.path.join(self.path, 'labels', idx + '.txt')

        img = cv2.imread(image_path)
        img_h, img_w, _ = img.shape

        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        label_small = np.ones(shape=[self.SMALL_SIZE, self.SMALL_SIZE, 3, 5])
        label_mid = np.ones(shape=[self.MID_SIZE, self.MID_SIZE, 3, 5])
        label_large = np.ones(shape=[self.LARGE_SIZE, self.LARGE_SIZE, 3, 5])

        label_small[...,0] = 0
        label_mid[..., 0] = 0
        label_large[..., 0] = 0
        label_file = open(label_path)

        aug = np.random.randint(0, len(self.AUGMENTS),1)[0]
        aug = self.AUGMENTS[aug]

        flip = np.random.choice([True, False], size=1)[0]
        print(flip, aug)

        img = self.augment_img(aug, img.copy(), FLIP=flip)

        for line in label_file:
            coord = np.array(line.strip().split()).astype(float)
            h, w, x, y = coord[0] * 2, coord[1] * 2, coord[3], coord[4]
            x = x / img_w * self.IMG_SIZE
            y = y / img_h * self.IMG_SIZE
            w = w / img_w * self.IMG_SIZE
            h = h / img_h * self.IMG_SIZE
            _box = np.array([x, y, w, h])


            box = self.augment_box(_box.copy(), FLIP=flip)
            best_anchor_idx = self.best_iou_anchor(box)

            if best_anchor_idx in [0,1,2]:
                grid_size = self.LARGE_SIZE
            elif best_anchor_idx in [3,4,5]:
                grid_size = self.MID_SIZE
            else:
                grid_size = self.SMALL_SIZE
            xidx, yidx = self.get_cell(grid_size, box[0], box[1])

            if best_anchor_idx in [0,1,2]:
                # for large grid, small object
                label_large[yidx, xidx, best_anchor_idx,0] = 1
                label_large[yidx, xidx, best_anchor_idx,1:] = box
            elif best_anchor_idx in [3,4,5]:
                # for medium grid, medium object
                label_mid[yidx, xidx, best_anchor_idx-3,0] = 1
                label_mid[yidx, xidx, best_anchor_idx-3,1:] = box
            else:
                # for small grid, large object
                label_small[yidx, xidx, best_anchor_idx-6,0] = 1
                label_small[yidx, xidx, best_anchor_idx-6,1:] = box


        return img, label_small, label_mid, label_large

    def best_iou_anchor(self, box):
        """
        :param box: xcenter, ycenter, w, h
        :return:
        """
        xcenter, ycenter, w, h = box

        box1 = (xcenter - w / 2, ycenter - h / 2, xcenter + w / 2, ycenter + h / 2)
        idx = 0
        best_iou = 0.

        for i in range(len(self.anchors)):
            anchor = self.anchors[i]
            anchor_w, anchor_h = anchor
            box2 = (xcenter - w / 2, ycenter - h / 2, xcenter + anchor_w / 2, ycenter + anchor_h / 2)
            current_iou = iou(box1, box2)
            if current_iou >= best_iou:
                best_iou = current_iou
                idx = i
        return idx

    def get_cell(self, grid_size, xcenter, ycenter):
        px_per_cell = float(self.IMG_SIZE) / grid_size
        x_idx = int((xcenter-1) / px_per_cell)
        y_idx = int((ycenter-1) / px_per_cell)
        return x_idx, y_idx

    def get(self, batch_size):
        images = []
        labels_small = []
        labels_mid = []
        labels_large = []
        for i in range(batch_size):

            idx = self.idxes[self.cursor]
            img, label_small, label_mid, label_large = self.get_single_data(idx)
            images.append(img)

            labels_small.append(label_small)
            labels_mid.append(label_mid)
            labels_large.append(label_large)


            self.cursor += 1
            if self.cursor == self.m:
                self.cursor = 0
                self.epoch += 1

        arr = np.arange(batch_size)
        np.random.shuffle(arr)

        return np.array(images)[arr], np.array(labels_small)[arr], np.array(labels_mid)[arr], np.array(labels_large)[arr]

    def augment_img(self, aug, img, FLIP=False):
        if aug == 'IDENTITY':
            pass
        elif aug == 'COLOR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        elif aug == 'AVERAGE':
            img = cv2.blur(img, (5, 5))
        elif aug == 'GAUSSIAN':
            img = cv2.GaussianBlur(img,(5,5),0)
        elif aug == 'MEDIAN':
            img = cv2.medianBlur(img, 5)
        if FLIP:
            img = img[:, ::-1, :]
        return img

    def augment_box(self, box, FLIP=False):
        if FLIP:
            x, y, w, h = box
            new_x = self.IMG_SIZE - x
            box = np.array([new_x, y, w, h])
        return box


def iou(box1, box2):
    """
    :param box1: x_upper_left, y_upper_left, x_lower_right, y_lower_right
    :param box2: x_upper_left, y_upper_left, x_lower_right, y_lower_right
    :return:
    """
    xul1, yul1, xlr1, ylr1 = box1
    xul2, yul2, xlr2, ylr2 = box2

    xul = max(xul1, xul2)
    yul = max(yul1, yul2)

    xlr = min(xlr1, xlr2)
    ylr = min(ylr1, ylr2)

    s_intersection = max(0, (xlr - xul) * (ylr - yul))
    s1 = (xlr1 - xul1) * (ylr1 - yul1)
    s2 = (xlr2 - xul2) * (ylr2 - yul2)

    return s_intersection / (s1 + s2  - s_intersection)
def plot_img_with_boxes(img, boxes):
    """
    :param img:
    :param boxes: a list of boxes [xcenter, ycenter, w, h]
    :return:
    """
    fig, ax = plt.subplots(1)
    ax.imshow(img.astype(int))
    for box in boxes:
        xcenter, ycenter, width, height = box
        x1, y1 = xcenter - width / 2, ycenter - height / 2
        x2, y2 = xcenter + width / 2, ycenter + height / 2

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
def prepare_data(path):
    folds = ['01','02','03','04','05','06','07','08','09','10']
    # folds = ['01']
    for fold in folds:
        idx_filename = os.path.join(path, 'FDDB-folds', 'FDDB-fold-' + fold + '.txt')
        label_filename = os.path.join(path, 'FDDB-folds', 'FDDB-fold-' + fold + '-ellipseList.txt')

        idx_file = open(idx_filename)
        label_file = open(label_filename)

        # for each picture
        for line in idx_file:


            idx = line.strip()

            new_label_file_name = os.path.join(path, 'labels', idx + '.txt')
            if not os.path.exists(os.path.dirname(new_label_file_name)):
                os.makedirs(os.path.dirname(new_label_file_name))
            new_label_file = open(new_label_file_name, 'w+')

            _ = label_file.readline().strip()
            num_faces = int(label_file.readline().strip())


            for i in range(num_faces):
                coord = label_file.readline().strip()
                new_label_file.write(coord + '\n')
def plot_label(img, label_small, label_mid, label_large):
    labels = [label_small, label_mid, label_large]
    boxes = []

    for label in labels:
        confidence = label[:,:,:,0]
        Is, Js, Ks = np.where(confidence>=0.5)

        n_boxes = len(Is)
        for i in range(n_boxes):
            box = label[Is[i], Js[i], Ks[i], 1:]
            boxes.append(box)

    plot_img_with_boxes(img, boxes)


if __name__ == '__main__':
    path = '../../../DL_data/FDDB/'
    # prepare_data(path)
    fddb = FDDBDataSet()
    n = 100

    for i in range(n):
        imgs, labels_small, labels_mid, labels_large = fddb.get(1)
        plot_label(imgs[0], labels_small[0], labels_mid[0], labels_large[0])








