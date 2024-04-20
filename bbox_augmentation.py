import cv2
import os
import albumentations as A
import uuid

"""
Bounding Box Augmentation Framework

Description:
This scripts augments YOLO format detection datasets based on Albumentations transformations

Author:
- Tanaz Ghahremani <tanaz.ghahremani@gmail.com>

"""


def save_bbox_file(file, bbox):
    """save_bbox_file saves yolo format bboxes to txt file correspondent to image file
    
    Args:
        file: file directory + file name
        bbox: nd list of bboxes (yolo)
    """
    f = open(file, "w")
    for i in range(len(bbox)):
        f.write(bbox[i][4] + " " + str(bbox[i][0]) + " " +
                str(bbox[i][1]) + " " + str(bbox[i][2]) + " " +
                str(bbox[i][3]) + "\n")
    f.close()


def do_aug_bbox(transformation, image_addr, image_labels, file_name, save_path):
    """augments image by given bboxes and save new image and labels
    
    Args:
        image_addr: directory of image
        imgage_labels: labels and bboxes in txt file (yolo)
        file_name: name of image and correspondent txt file
        save_path: directory of new image and label
    """
    image = cv2.imread(image_addr)
    if not os.path.isfile(image_labels):
        return
    f = open(image_labels, "r")
    bb = []
    if os.stat(image_labels).st_size == 0:
        return
    for line in f:
        line = line.split(" ")
        t = line[0]
        line = line[1:]
        temp = list(map(float, line))
        # print(temp)
        temp.append(t)
        if (1 >= temp[0] >= 0) and (1 >= temp[1] >= 0) and temp[2] != 0.0 and temp[3] != 0.0:
            bb.append(temp)
    
    transformed = transformation(image=image, bboxes=bb)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    id = uuid.uuid4().hex
    cv2.imwrite(os.path.join(save_path + "/images", file_name + "_" + id + ".jpg"), transformed_image)
    save_bbox_file(save_path + "/labels/" + file_name + "_" + id + ".txt", transformed_bboxes)


#Usage
folder = "Data/Dataset_Noor_Ind_aug/"
images = "train/images/"
labels = "train/labels/"
# file_name = "000000000074"
for filename in os.listdir(folder + images):
    name = filename[:-4]
    print(name)
    # if name == file_name:
    #     continue
    v_flip = A.Compose([A.VerticalFlip(p=1)],
                       bbox_params=A.BboxParams(format='yolo', min_visibility=1))
    do_aug_bbox(v_flip, folder + images + filename,
                folder + labels + name + ".txt", name, "Data/Dataset_Noor_Ind_aug/aug")
