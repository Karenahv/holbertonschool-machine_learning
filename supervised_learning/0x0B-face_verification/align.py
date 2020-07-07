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

    def find_landmarks(self, image, detection):
        """finds facial landmarks"""
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype="int")
        # Find landmark points in image
        landmark = self.shape_predictor(image, detection)
        if landmark:
            for i in range(0, 68):
                coords[i] = (landmark.part(i).x, landmark.part(i).y)
        else:
            coords = None
        return coords

    def align(self, image, landmark_indices, anchor_points, size=96):
        """aligns an image for face verification"""
        # Detect face in image and find landmarks
        box = self.detect(image)
        landmarks = self.find_landmarks(image, box)

        # Select three points in the landmarks(Eyes and nose)
        points_in_image = landmarks[landmark_indices]
        points_in_image = points_in_image.astype('float32')
        # Generate the normalized output size
        output_size = anchor_points * size

        # Calculates the 2 \times 3 matrix of an affine transform
        affine_transf = cv2.getAffineTransform(points_in_image, output_size)

        # Transforms the source image using the specified matrix
        transformed_img = cv2.warpAffine(image, affine_transf, (size, size))

        return transformed_img
