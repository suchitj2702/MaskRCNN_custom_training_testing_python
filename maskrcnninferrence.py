# @Author: Suchit Jain
# @Date:   2018-08-14T08:54:58+05:30
# @Email:  suchit27022@gmail.com
# @Last modified by:   Suchit Jain
# @Last modified time: 2018-10-01T18:12:44+05:30
# @License: Free

""" Usage: Run from command line as such(recommended python 3.6.5):

    # Using trained coco weights for inferrence
    python maskrcnninferrence.py coco

    # Using the last saved weights by maskrcnntrain for inferrence
    python maskrcnninferrence.py last

    # Using custom trained weights for inferrence
    python maskrcnninferrence.py custom
"""
import cv2
import numpy as np
import maskrcnntrain
from mrcnn import utils
from mrcnn import model as modellib
import os
import json
import pandas as pd
import coco

LOGS_FOLDER = "logs"
PRETRAINED_COCO_WEIGHTS = "mask_rcnn_coco.h5"
TRAINED_MASKRCNN_WEIGHTS = "trainedweights.h5"
CLASSES = 'class1' # use a list of classes if multiple classes
IMG_DIR = "TestImages/Images"
INFERRED_DIR = "TestImages/InferredImages"

class InferenceConfigtrained(maskrcnntrain.trainingconfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9 # Skip detections with < 90 % confidence

class InferenceConfigcoco(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_weights():
    global class_names, weights_path, model
    if WEIGHTS_TO_USE == "coco":
        config = InferenceConfigcoco()
        config.display()
        model = modellib.MaskRCNN(
            mode = "inference", model_dir = LOGS_FOLDER, config = config
        )
        weights_path = PRETRAINED_COCO_WEIGHTS
        model.load_weights(weights_path, by_name=True)
        class_names = [
            'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
            'teddy bear', 'hair drier', 'toothbrush'
        ]
    elif WEIGHTS_TO_USE == "last":
        config = InferenceConfigtrained()
        config.display()
        model = modellib.MaskRCNN(
            mode = "inference", model_dir = LOGS_FOLDER, config = config
        )
        weights_path = model.find_last()[1]
        model.load_weights(weights_path, by_name=True)
        class_names = CLASSES

    else:
        config = InferenceConfigtrained()
        config.display()
        model = modellib.MaskRCNN(
            mode = "inference", model_dir = LOGS_FOLDER, config = config
        )
        weights_path = TRAINED_MASKRCNN_WEIGHTS
        model.load_weights(weights_path, by_name=True)
        class_names = CLASSES

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha = 0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha *c,
            image[:, :, n]
        )
    return image

def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and label
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{}{:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image =cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

    return image, n_instances

def main():
    for filename in os.listdir(IMG_DIR):
        test_image = cv2.imread(os.path.join(IMG_DIR,filename))
        results = model.detect([test_image], verbose = 0)
        r = results[0]
        inferred, number_of_instances = display_instances(
            test_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        cv2.imwrite(os.path.join(INFERRED_DIR,filename), inferred)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('weights_to_use',
                        metavar = "<weights_to_use>",
                        help = 'Can either be coco, last or custom')
    args = parser.parse_args()
    WEIGHTS_TO_USE = args.weights_to_use
    load_weights()
    colors = random_colors(len(class_names))
    class_dict = {name: color for name, color in zip(class_names, colors)}
    main()
