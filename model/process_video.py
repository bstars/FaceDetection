import sys
sys.path.append('..')

import cv2
import numpy as np
from model.Detector import Detector

import util.config as cfg

def plot_img_with_boxes_cv(img, boxes):
    """
        :param img:
        :param boxes: a list of boxes [xcenter, ycenter, w, h]
        :return:
        """

    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float) / 255
    image = img.astype(np.float) / 255

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
    out = cv2.VideoWriter('result2.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    for img in images:
        out.write(img)
    out.release()

if __name__ == '__main__':
    detector = Detector()
    vidcap = cv2.VideoCapture('../eg1.mp4')
    labeled_images = []
    success,image = vidcap.read()
    count = 3000

    while success and count < 5000:
        print(count)
        image = cv2.resize(image, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        boxes = detector.predict(image)
        labeled_img = plot_img_with_boxes_cv(image, boxes)
        print(labeled_img.shape)
        cv2.imshow('',labeled_img)
        cv2.waitKey(0)

        labeled_images.append(labeled_img)
        success,image = vidcap.read()
        count += 1

    generate_video(labeled_images)



