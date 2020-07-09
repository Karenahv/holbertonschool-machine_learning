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


def save_images(path, images, filenames):
    """save images"""
    try:
        os.chdir(path)
        for name, img in zip(filenames, images):
            cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        os.chdir('../')
        return True

    except FileNotFoundError:
        return False


def generate_triplets(images, filenames, triplet_names):
    """ generate triplets images, true, true, false"""
    list_A = []
    list_P = []
    list_N = []
    _, h, w, c = images.shape
    names = [[name[0]+'.jpg', name[1]+'.jpg',
               name[2]+'.jpg'] for name in triplet_names]

    for name in names:
        flagA, flagP, flagN = (0, 0, 0)

        A_name, P_name, N_name = name

        if A_name in filenames:
            flagA = 1
        if P_name in filenames:
            flagP = 1
        if N_name in filenames:
            flagN = 1

        if flagA and flagP and flagN:
            index_A = filenames.index(A_name)
            index_P = filenames.index(P_name)
            index_N = filenames.index(N_name)

            A = images[index_A]
            P = images[index_P]
            N = images[index_N]

            list_A.append(A)
            list_P.append(P)
            list_N.append(N)

    list_A = [elem.reshape(1, h, w, c) for elem in list_A]
    list_A = np.concatenate(list_A)

    list_P = [elem.reshape(1, h, w, c) for elem in list_P]
    list_P = np.concatenate(list_P)

    list_N = [elem.reshape(1, h, w, c) for elem in list_N]
    list_N = np.concatenate(list_N)

    return (list_A, list_P, list_N)
