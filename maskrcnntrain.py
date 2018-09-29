import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.config import Config
from mrcnn import model as modellib, utils

PRETRAINED_COCO_WEIGHTS = "mask_rcnn_coco.h5"
WEIGHTS_TO_USE = "coco" # Can be either coco or last(for continuing training)
LOGS_FOLDER = "logs"
DATASET_FOLDER = "datasetnew"
CLASS = "building/plot"

class trainingconfig(Config):
    """
    Derives from the base Config class and overrides some values
    """
    NAME = "object_detector"

    IMAGES_PER_GPU = 2

    NUM_CLASSES = 2 # For single class classification(including background)

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.9 # Skip detections with < 90 % confidence

class Dataset(utils.Dataset):
    def load_dataset(self, dataset_dir, subset):
        self.add_class(CLASS, 1, CLASS)

        assert subset in ["train", "val"]
        dataset_dir = dataset_dir + "/" + subset
        annotations = json.load(open(dataset_dir + "/" + "datasetroad.json"))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            image_path = dataset_dir + "/" + a['filename']
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                CLASS,
                image_id = a['filename'],
                path = image_path,
                width = width,
                height = height,
                polygons = polygons)

    def load_mask(self, image_id):
        # If not an dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != CLASS:
            return super(self.__class__, self).load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instanc_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])]
                        ,dtype = np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype = np.int32)

    def image_reference(self, image_id):
        """Return the path of the image"""
        info = self.image_info[image_id]
        if info["source"] == CLASS:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model"""
    # Training dataset
    dataset_train = Dataset()
    dataset_train.load_dataset(DATASET_FOLDER, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Dataset()
    dataset_val.load_dataset(DATASET_FOLDER, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate = config.LEARNING_RATE,
                epochs = 30,
                layers = 'heads')


if __name__ == '__main__':
    # Configuration for the training
    config = trainingconfig()
    # Create Model
    model = modellib.MaskRCNN(mode = "training", config = config, model_dir = LOGS_FOLDER)

    # Load Pretrained Weights
    if WEIGHTS_TO_USE == "coco":
        weights_path = PRETRAINED_COCO_WEIGHTS
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        weights_path = model.find_last()[1]
        model.load_weights(weights_path, by_name=True)

    train(model)
