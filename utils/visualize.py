import cv2
import os
import numpy as np


def visualise_bbox(img, pts, width, height, colors):
    """cv2 image show with bounding box and class
    
    Args:
        img: cv2 image
        pts: nd list of bounding box coordinations + class
        width: image width
        height: image height
        colors: class colors (random)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(pts)):
        end_point = (int(round((float(pts[i][0])+float(pts[i][2])/2) * width)),
                     int(round((float(pts[i][1])+float(pts[i][3])/2) * height)))
        start_point = (int(round((float(pts[i][0])-float(pts[i][2])/2) * width)),
                       int(round((float(pts[i][1])-float(pts[i][3])/2) * height)))
        color = tuple(map(int, colors[int(pts[i][4])]))
        cv2.rectangle(img, start_point, end_point, color=color, thickness=1)
        cv2.putText(img, str(pts[i][4]), end_point, font, 1, color, 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)  # press any key


def visualise_polygon(img, pts, labels, colors):
    """cv2 image show with polygon and class
    
    Args:
        img: cv2 image
        pts: nd list of polygon coordinations
        labels: list of polygons classes
        colors: class colors (random)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, mask_pts in enumerate(pts):
        mask_pts = np.array(mask_pts, np.int32)
        mask_pts = mask_pts.reshape((-1, 1, 2))
        color = tuple(map(int, colors[int(labels[i])]))
        img = cv2.polylines(img, [mask_pts], True, color=color, thickness=3)
        cv2.putText(img, labels[i], mask_pts[0][0], font, 1, color, 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)  # press any key


def detection_visualize(folder, images, labels, colors):
    """visualizes detection datasets
    
    Args:
        folder: root directory of dataset
        images: directory of images
        labels: directory of labels
        colors: class colors (random)
    """
    for filename in os.listdir(folder + images):
        # you can also filter images
        
        # if re.search("^a", filename):
        #     continue
        # print(filename)
        # name = ""
        # if filename != name:
        #     continue
        
        image_path = folder + images + filename
        filename = filename[:-4]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(filename)
        if not os.path.isfile(folder + labels + filename + ".txt"):
            continue
        f1 = open(folder + labels + filename + ".txt", "r")
        Lines = f1.readlines()
        p = []
        for l in Lines:
            l = l.split()
            new_l = list(l[1:])
            new_l.insert(4, l[0])
            p.append(new_l)
        p = np.array(p)
        visualise_bbox(img, p, img.shape[1], img.shape[0], colors)


def segmentation_visualize(folder, images, labels, colors):
    """visualizes segmentation datasets
    
    Args:
        folder: root directory of dataset
        images: directory of images
        labels: directory of labels
        colors: class colors (random)
    """
    # name = ""
    for filename in os.listdir(folder + images):
        # if filename != name:
        #     continue
        print(filename)
        image_path = folder + images + filename
        filename = filename[:-4]
        img = cv2.imread(image_path)
        if not os.path.isfile(folder + labels + filename + ".txt"):
            continue
        f1 = open(folder + labels + filename + ".txt", "r")
        Lines = f1.readlines()
        p = []
        label = []
        for l in Lines:
            # print(l)
            label.append(l.split()[0])
            temp = list(l.split()[1:])
            # print(temp)
            temp2 = []
            for i in range(0,len(temp)-1,2):
                temp2.append([int(round(float(temp[i])*img.shape[1])), int(round(float(temp[i+1])*img.shape[0]))])
            p.append(temp2)
        visualise_polygon(img, p, label, colors)


#Usage
images = 'images/train/'
labels = 'labels/train/'
folder = "test_data/"
class_number = 80
colors = []
for i in range(class_number):
    colors.append(tuple(np.random.choice(range(256), size=3)))
    
segmentation_visualize(folder, images, labels, colors)    