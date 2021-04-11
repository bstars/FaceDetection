# [x, y, w, h]

IMG_SIZE = 448
ANCHORS = [
    (10, 13), (16, 30), (33, 23),       # small anchor box for small objects, large-sized labels
    (30, 61), (62, 45), (59, 119),      # medium anchor box for medium-sized objects, medium-sized labels
    (116, 90), (156, 198), (373, 326)   # large anchor box for large objects, small-sized labels
]
# These anchor boxes are from YOLO V3 but are pretty compatible with the face sizes in the dataset

SMALL_SIZE = 14
MID_SIZE = 28
LARGE_SIZE = 56

VGG_PATH = '../../../DL_data/vgg16/vgg16_weights.npz'
FDDB_PATH = '../../../DL_data/FDDB'
CKPT_PATH = '../ckpt4/model.ckpt'
TEST_DATA_PATH = '../test_imgs'

# Training parameter
BATCH_SIZE = 2
LEAKY_RELU_ALPHA = 0.1
BATCH_NORM_MOMENTUM = 0.9
BATCH_NORM_EPS = 1e-4
REG = 0.0005

OBJECT_SCALE = 50.
NOOBJECT_SCALE = 0.5
COORD_SCALE = 20.

SCALE_SMALL_OBJS = 1.
SCALE_MID_OBJS = 1.
SCALE_LARGE_OBJS = 1.


# Testing parameter
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.2

INPUT_NAME = 'images:0'
TRAINING_NAME = 'training:0'
DETECT1_NAME = 'yolo_v3/Reshape:0'
DETECT2_NAME = 'yolo_v3/Reshape_1:0'
DETECT3_NAME = 'yolo_v3/Reshape_2:0'
