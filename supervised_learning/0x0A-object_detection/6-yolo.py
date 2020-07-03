#!/usr/bin/env python3
""" class Yolo"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2


def iou_cajas(boxa, boxb):
    """Funcion de calculo de la Intersección
     sobre la Union de las Cajas o IOU """
    intx1 = max(boxa[0], boxb[0])
    inty1 = max(boxa[1], boxb[1])
    intx2 = min(boxa[2], boxb[2])
    inty2 = min(boxa[3], boxb[3])

    intarea = max(0, (intx2 - intx1)) * max(0, (inty2 - inty1))
    boxaarea = (boxa[2] - boxa[0]) * (boxa[3] - boxa[1])
    boxbarea = (boxb[2] - boxb[0]) * (boxb[3] - boxb[1])
    return intarea / (boxaarea + boxbarea - intarea)


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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Perform non-max suppression on the boundary boxes.
        """
        sort_order = np.lexsort((-box_scores, box_classes))
        box_scores = box_scores[sort_order]
        box_classes = box_classes[sort_order]
        filtered_boxes = filtered_boxes[sort_order]
        del_idxs = []
        for idx in range(len(box_scores)):
            if idx in del_idxs:
                continue
            clas = box_classes[idx]
            box = filtered_boxes[idx]
            for cidx in range(idx + 1, len(box_scores)):
                if (box_classes[cidx] != clas):
                    break
                if ((iou_cajas(filtered_boxes[cidx], box)
                     >= self.nms_t)):
                    del_idxs.append(cidx)
        return (np.delete(filtered_boxes, del_idxs, axis=0),
                np.delete(box_classes, del_idxs),
                np.delete(box_scores, del_idxs))

    @staticmethod
    def load_images(folder_path):
        """load images"""
        images = []
        image_paths = glob.glob(folder_path + '/*', recursive=False)

        # creating the images list
        for imagepath_i in image_paths:
            images.append(cv2.imread(imagepath_i))

        return(images, image_paths)

    def preprocess_images(self, images):
        """
            Resize the images with inter-cubic interpolation
            Rescale all images to have pixel values in the range [0, 1]
        """

        dims = []
        res_images = []

        input_h = self.model.input.shape[1].value
        input_w = self.model.input.shape[2].value
        for image in images:
            dims.append(image.shape[:2])

        dims = np.stack(dims, axis=0)

        newtam = (input_h, input_w)

        interpolation = cv2.INTER_CUBIC
        for image in images:
            resize_img = cv2.resize(image, newtam, interpolation=interpolation)
            resize_img = resize_img / 255
            res_images.append(resize_img)

        res_images = np.stack(res_images, axis=0)

        return (res_images, dims)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Show predicted boxes
        """
        rounded_scores = np.around(box_scores, decimals=2)

        for i, box in enumerate(boxes):
            x1, y2, x2, y1 = box

            start = (int(x1), int(y1))
            end = (int(x2), int(y2))
            blue = (255, 0, 0)

            cv2.rectangle(image, start, end, blue, 2)

            score = str(rounded_scores[i])
            class_name = self.class_names[box_classes[i]]
            classname_score = "{} {}".format(class_name, score)
            start_text = (int(x1), int(y2)-5)
            font_name = cv2.FONT_HERSHEY_SIMPLEX
            red = (0, 0, 255)
            text_thickness = 1
            line_type = cv2.LINE_AA
            font_scale = 0.5

            cv2.putText(image,
                        classname_score,
                        start_text,
                        font_name,
                        font_scale,
                        red,
                        text_thickness,
                        line_type)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists("./detections"):
                os.makedirs("./detections")
            cv2.imwrite("./detections/{}".format(file_name), image)
        cv2.destroyAllWindows()
