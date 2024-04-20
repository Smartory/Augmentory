import cv2
import os
import albumentations as A
import numpy as np
import uuid

"""
Polygon Augmentation Framework

Description:
This scripts augments YOLO format segmentation datasets based on Albumentations transformations

Author:
- Tanaz Ghahremani <tanaz.ghahremani@gmail.com>
- Mohammad Hoseini <mohammadhosini60@gmail.com>

"""


def save_poly_file(file, poly):
    """saves yolo format poly to txt file correspondent to image file
    
    Args:
        file: fie directory + file name
        poly: nd list of polygons (yolo)
    """
    f = open(file, "w")
    for i in range(len(poly)):
        for j in range(len(poly[i])):
            if j == (len(poly[i]) - 1):
                f.write(str(poly[i][j]))
            else:
                f.write(str(poly[i][j]) + " ")
        f.write("\n")
    f.close()


def area_from_txt_line(coords, width, height):
    """creates a cv2 polygon and calculate its area in pixels
    
    Args:
        coords: point coordinations of polygon (yolo)
        width: image width
        height: image height
    Return:
        area: area of polygon    
    """
    coords_x = coords[::2]
    coords_x *= width
    coords_y = coords[1::2]
    coords_y *= height
    polygon = np.column_stack((coords_x, coords_y))
    polygon = polygon.astype(np.int32)
    # Calculate the area of the polygon
    area = cv2.contourArea(polygon)
    return area


def do_aug_poly(transformation, image_addr, image_labels, file_name, save_path):
    """augment image with albumentations transformation by given polygon label and save new image and label
    
    Args:
        transformation: albumentations transformation(all applicable except dropout)
        image_addr: directory of image
        image_labels: directory of labels and polygons in txt file (yolo)
        file_name: name of image and correspondent txt file
        save_path: directory of new image and label
    """
    image = cv2.imread(image_addr)

    key_point, class_labels = YOLO2KeyPoint(image_labels, image)

    transformed = transformation(image=image, keypoints=key_point, class_labels=class_labels)
    transformed_image = transformed['image']
    transformed_keypoints = transformed['keypoints']
    transformed_class_labels = transformed['class_labels']

    poly = keyPoint2YOLO(transformed_keypoints, transformed_class_labels,
                         transformed_image.shape[1], transformed_image.shape[0])
    new_id = uuid.uuid4().hex
    cv2.imwrite(os.path.join(save_path + "/images", file_name + "_" + new_id + ".jpg"), transformed_image)
    save_poly_file(save_path + "/labels/" + file_name + "_" + new_id + ".txt", poly)


def YOLO2KeyPoint(image_labels, image):
    """convert yolo format polygon coordinations to keypoints

    Args:
        image_labels: directory of labels and polygons in txt file (yolo)
        image: cv2 image
    Return:
        key_point: list of polygons key points ([x, y] format in pixels)
        class_labels: list of coressponding labels of key points (id_label_area for each key point)
    """
    if not os.path.isfile(image_labels):
        return
    f = open(image_labels, "r")
    key_point = []
    class_labels = []
    for id, line in enumerate(f):
        line = line.split(" ")
        label = line[0]
        line = line[1:]
        temp = list(map(float, line))
        area = area_from_txt_line(np.array(temp), image.shape[1], image.shape[0])
        for i in range(0, len(temp) - 1, 2):
            # Label: id_label_area
            class_labels.append(str(id) + "_" + label + "_" + str(area))
            x = int(temp[i] * image.shape[1])
            y = int(temp[i + 1] * image.shape[0])
            if x >= image.shape[1]:
                x = image.shape[1] - 1
            if y >= image.shape[0]:
                y = image.shape[0] - 1
            key_point.append((x, y))
    return key_point, class_labels


def keyPoint2YOLO(key_point_list, class_label_list, width, height, overlap=0.2):
    """converts keypoints to yolo format coordinations
     
    Args:
        key_point_list: list of keypoints
        class_label_list: list of labels
        width: width of image
        height: height of image
    Return:
        poly: nd list of polygons (yolo)
    """
    poly = []
    line = []
    pre_id = -1
    pre_label = -1
    for i, class_label in enumerate(class_label_list):
        class_label = class_label.split("_")
        id = class_label[0]
        label = class_label[1]
        original_area = float(class_label[2])
        if pre_id != id and pre_id != -1:
            area = area_from_txt_line(np.array(line), width, height)
            if (area / original_area) >= overlap:
                line.insert(0, pre_label)
                poly.append(line)
            line = []
        pre_id = id
        pre_label = label
        line.append(key_point_list[i][0] / width)
        line.append(key_point_list[i][1] / height)
        if i == (len(class_label_list) - 1):
            area = area_from_txt_line(np.array(line), width, height)
            if (area / original_area) >= overlap:
                line.insert(0, pre_label)
                poly.append(line)
    return poly


# Usage
folder = "test/"
images = "images/train2017/"
labels = "labels/train2017/"
# file_name = "000000000074"
for filename in os.listdir(folder + images):
    name = filename[:-4]
    print(name)
    # if name == file_name:
    #     continue
    v_flip = A.Compose([A.VerticalFlip(p=1)],
                       keypoint_params=A.KeypointParams(format='xy'))
    do_aug_poly(v_flip, folder + images + filename,
                folder + labels + name + ".txt", name, "test/results")
 
