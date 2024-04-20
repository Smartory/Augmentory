# Augmentory

## Description
This repository consists of detection and segmentation dataset augmentation frameworks for YOLO format datasets based on Albumentations transformations.

## Author
- **Tanaz Ghahremani** <tanaz.ghahremani@gmail.com>
- **Mohammad Hoseini** <mohammadhosini60@gmail.com>

## 🚀 How to Use
To effectively utilize the Polygon/Mask/BBox Augmentation Framework, please follow these steps:

1. Environment Setup: Confirm that Python and all required dependencies are properly installed on your system.
2. Data Organization: Place your image files in the `images/train/` directory and their corresponding label files in `labels/train/`. This will facilitate easy access and processing by the script.
3. Configuration: Modify the folder, images, and labels variables within the script to align with your specific directory structure. This ensures that the script accurately locates and processes your files.
4. Execution: Run the script to initiate the augmentation process. Augmented images and labels will be systematically generated and stored in the `test_data/results/` directory.

+ Note that `result` directory should consists of `images` and `labels` subdirectories.

By adhering to these structured steps, you can seamlessly augment your datasets to enhance your machine learning model's performance.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- Albumentations (`albumentations`)
- NumPy (`numpy`)

## Modification
1. You can use any transformations other than albumentations
2. Create your transformation (Dropouts are not applicable!).
3. Modify and apply it on `do_aug_poly`.
4. You can also change the save policy (YOLOv8 is default).
5. You can also change "Overlap Policy" (Polygon Area is default)

## Utils
1. `COCOtoYOLO`: It converts COCO JSON format labels to YOLO format
2. `poly_mask_conversion`: It converts polygons to masks and vice versa.
3. `visualize`: It visualize bounding boxes, polygons and coressponding datasets.

## Contributing
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

## Paper
For more information and results analysis, please check out our paper: "Paper name and link"

