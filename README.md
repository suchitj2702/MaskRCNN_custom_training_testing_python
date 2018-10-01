# Mask R-CNN python implementation for training and testing custom datasets
This implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras and Tensorflow is a simplified version of the [matterport Mask_RCNN](https://github.com/matterport/Mask_RCNN) implementation. This implementation allows the user to train and test on custom datasets, by following some basic and specific dataset structuring.

The [training](maskrcnntrain.py) and [testing](maskrcnninferrence.py) code has cues from the [matterport Mask_RCNN](https://github.com/matterport/Mask_RCNN) with some custom changes to allow easy structuring of dataset and the ability to train on custom multiclass datasets
