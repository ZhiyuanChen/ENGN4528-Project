import cv2
import numpy as np
from obstacle.mrcnn import coco
from obstacle.mrcnn import utils
import obstacle.mrcnn.model as modellib
from obstacle.mrcnn import visualize
from obstacle.mrcnn.model import log
from obstacle.mrcnn.config import Config



class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def draw_result(image, result, show_mask=True, show_bbox=True):
    coordinates = result['rois']
    masks = result['masks']
    class_ids = result['class_ids']
    scores = result['scores']
    colors = color_map()
    for i in range(coordinates.shape[0]):
        color = colors[class_ids[i]].astype(np.int).tolist()
        if show_bbox:
            coordinate = coordinates[i]
            cls = class_names[class_ids[i] - 1]
            score = scores[i]
            cv2.rectangle(image, (coordinate[1], coordinate[0]), (coordinate[3], coordinate[2]), color, 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, '{}: {:.3f}'.format(cls, score), (coordinate[1], coordinate[0]),
                        font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        if show_mask:
            mask = masks[:, :, i]
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.int)
            color_mask[mask] = color
            image = cv2.addWeighted(color_mask, 0.5, image.astype(np.int), 1, 0)
    return image
