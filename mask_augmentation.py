import cv2
import os
import albumentations as A
from utils.poly_mask_conversion import poly2mask, mask2poly

"""
Mask Augmentation Framework

Description:
This scripts augments YOLO format segmentation datasets based on Albumentations transformations. It first
converts polygons to masks, applies the transformation and then converts masks back to polygons. 
It can be also use as Mask Augmentation (No polygon-mask conversion)

Author:
- Tanaz Ghahremani <tanaz.ghahremani@gmail.com>

"""


folder = "test/"
images = "images/train2017/"
labels = "labels/train2017/"
# file_name = "000000000074"
for filename in os.listdir(folder + images):
    name = filename[:-4]
    print(name)
    # if name == file_name:
    #     continue
    img = cv2.imread(folder + images + filename)
    # Converts Polygon to Mask
    masks, ls = poly2mask(folder, name, img, labels, folder)

    v_flip = A.Compose([A.VerticalFlip(p=1)])

    transformed = v_flip(image=img, masks=masks)
    transformed_image = transformed['image']
    transformed_masks = transformed['masks']

    cv2.imwrite(os.path.join('test/results' + "/images", filename), transformed_image)
    # Converts Mask to Polygon
    mask2poly(folder, name, img, 'results/labels/', transformed_masks, ls)


