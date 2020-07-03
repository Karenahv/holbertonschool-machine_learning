#!/usr/bin/env python3
""" class Yolo"""

import tensorflow.keras as K
import numpy as np


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

    def _sigmoidal(self, x):
        """
        sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process Outputs
        """
        boxes = []
        for i in range(len(outputs)):
            tx, ty, _, _ = np.split(outputs[i], (2, 4, 5), axis=-1)
            grid_size = np.shape(outputs[i])[1]
            C_xy = np.meshgrid(range(grid_size), range(grid_size))

            C_xy = np.stack(C_xy, axis=-1)

            C_xy = np.expand_dims(C_xy, axis=2)

            b_xy = self._sigmoidal(tx) + C_xy

            b_xy = b_xy / grid_size

            inp = self.model.input_shape[1:3]
            b_wh = (np.exp(ty) / inp) * self.anchors[i]

            bx = b_xy[:, :, :, :1]
            by = b_xy[:, :, :, 1:2]
            bw = b_wh[:, :, :, :1]
            bh = b_wh[:, :, :, 1:2]

            x1 = (bx - bw / 2) * image_size[1]
            y1 = (by - bh / 2) * image_size[0]
            x2 = (bx + bw / 2) * image_size[1]
            y2 = (by + bh / 2) * image_size[0]

            boxes.append(np.concatenate([x1, y1, x2, y2], axis=-1))
        box_confidence = [self._sigmoidal(out[..., 4:5]) for out in outputs]
        box_prob = [self._sigmoidal(out[..., 5:]) for out in outputs]

        return boxes, box_confidence, box_prob

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes
        """
        all_boxes = np.concatenate([boxs.reshape(-1, 4) for boxs in boxes])
        class_probs = np.concatenate([probs.reshape(-1,
                                                    box_class_probs[0].
                                                    shape[-1])
                                      for probs in box_class_probs])
        all_classes = class_probs.argmax(axis=1)
        all_confidences = (np.concatenate([conf.reshape(-1)
                                           for conf in box_confidences])
                           * class_probs.max(axis=1))
        thresh_idxs = np.where(all_confidences < self.class_t)
        return (np.delete(all_boxes, thresh_idxs, axis=0),
                np.delete(all_classes, thresh_idxs),
                np.delete(all_confidences, thresh_idxs))
