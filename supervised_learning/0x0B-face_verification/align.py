#!/usr/bin/env python3
""" utils functons fo face recognition"""

import numpy as np
import dlib
import cv2


class FaceAlign:
    """class face align"""

    def __init__(self, shape_predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """detects a face in an image and
        return rectangle face"""

        max = 0
        try:
            rects = self.detector(image, 1)
            if len(rects) == 0:
                rectangle = dlib.rectangle(0, 0, image.shape[1],
                                           image.shaep[0])
            else:
                for (i, rect) in enumerate(rects):
                    if rect.area() > max:
                        rectangle = rect
            return rectangle
        except RuntimeError:
            return None
