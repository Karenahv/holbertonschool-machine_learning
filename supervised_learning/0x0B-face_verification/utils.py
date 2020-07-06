#!/usr/bin/env python3
""" utils functons fo face recognition"""

import os
import cv2
import numpy as np
import csv


def load_images(images_path, as_array=True):
    """
    Load images from a folder. Return as ndarray or list
    """
    file_list = os.listdir('./HBTN')
    images = []
    file_names = []
    for file in sorted(file_list):
        path = images_path + '/' + file
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        file_names.append(file)
    if as_array:
        images = np.asarray(images)
    return images, file_names


def load_csv(csv_path, params={}):
    """ loads the content of a csv file as
        a lists of list"""
    csv_values = []
    with open(csv_path, encoding="utf-8") as file:
        sreader = csv.reader(file, params)
        # Reads all lines of file
        for row in sreader:
            csv_values.append(row)
    return csv_values
