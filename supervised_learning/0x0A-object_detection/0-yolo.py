#!/usr/bin/env python3
""" class Yolo"""

import tensorflow.keras as K


class Yolo:
    """performs object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initialize class"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
