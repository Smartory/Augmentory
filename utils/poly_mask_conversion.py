import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import supervision as sv


def poly2mask(folder_path, filename, image, poly_label, save_path="", save_masks=False):
    """converts polygons in yolo format to masks
     
    Args:
        folder_path: root directory of dataset
        filename: name of image and correspondent txt file
        image: cv2 image
        poly_label: directory of labels and polygons in txt file (yolo)
        save_path: directory of new masks 
        save_mask: True if you want to save masks
    Return:
        masks: nd list of masks
        labels: list of labels (for instance segmentation)
    """
    width = image.shape[1]
    height = image.shape[0]
    # print(image.shape)
    polygon_filename = folder_path + poly_label + filename + ".txt"
    if not os.path.isfile(polygon_filename):
        return
    polygon_file = open(polygon_filename, "r")
    polygons = polygon_file.readlines()
    masks = []
    labels = []
    for idx, polygon in enumerate(polygons):
        polygon = polygon.split(" ")
        label = polygon[0]
        labels.append(label)
        polygon = list(map(float, polygon[1:-2]))
        p = []
        for i in range(0, len(polygon) - 1, 2):
            p.append([int(polygon[i] * width), int(polygon[i + 1] * height)])
        mask = sv.polygon_to_mask(np.array(p), (width, height))
        masks.append(mask)
        if save_masks:
            cv2.imwrite(save_path + 'results/' + label + str(idx)+'.png', (mask * 255).astype(np.uint8))
    return masks, labels


def mask2poly(folder_path, filename, image, save_path, masks, labels, mask_path=""):
    """converts masks to polygons in yolo format and save in txt file
     
    Args:
        folder_path: root directory of dataset
        filename: name of image and correspondent txt file
        image: cv2 image
        save_path: directory of new polygons txt
        masks: nd list of masks
        labels: list of labels (for instance segmentation)
        mask_path: directory of saved masks (if you want to read masks from directory)
    """
    width = image.shape[1]
    height = image.shape[0]
    label = ""
    # print(image.shape)
    polygon_filename = folder_path + save_path + filename + ".txt"
    polygon_file = open(polygon_filename, "w")
    # if you want to read masks from directory uncomment the following comments and modify the code
    # based on your mask format
    # for mask in os.listdir(folder_path + mask_path):
    for i, mask in enumerate(masks):
    #     label = mask[:-4]
    #     m = cv2.imread(folder_path + mask_path + mask, cv2.COLOR_BGR2GRAY)
    #     m = np.uint8(m/255)
        poly = sv.mask_to_polygons(mask)
        polygon_file.write(str(labels[i]) + " ")
        for i in range(0, len(poly[0])):
            polygon_file.write(str(round(float(poly[0][i][0]/width),6)) + " " + str(round(float(poly[0][i][1]/height),6)) + " ")
        polygon_file.write("\n")
    return
