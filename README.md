# Augmentory
[![arXiv](https://img.shields.io/badge/arXiv-2405.04442-b31b1b.svg)](https://arxiv.org/abs/2405.04442) 
[![GitHub stars](https://img.shields.io/github/stars/Smartory/Augmentory.svg?style=social)](https://github.com/Smartory/Augmentory/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Smartory/Augmentory.svg?style=social)](https://github.com/Smartory/Augmentory/network)
[![GitHub issues](https://img.shields.io/github/issues/Smartory/Augmentory.svg)](https://github.com/Smartory/Augmentory/issues)
[![GitHub license](https://img.shields.io/github/license/Smartory/Augmentory.svg)](https://github.com/Smartory/Augmentory/blob/master/LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/Smartory/Augmentory.svg)](https://github.com/Smartory/Augmentory/commits/master)
[![GitHub contributors](https://img.shields.io/github/contributors/Smartory/Augmentory.svg)](https://github.com/Smartory/Augmentory/graphs/contributors)
[![GitHub repo size](https://img.shields.io/github/repo-size/Smartory/Augmentory.svg)](https://github.com/Smartory/Augmentory)
[![Visits Badge](https://badges.pufler.dev/visits/Smartory/Augmentory)](https://github.com/Smartory/Augmentory)

## Description
This repository consists of detection and segmentation dataset augmentation frameworks for YOLO format datasets based on Albumentations transformations.

## Authors
- **Tanaz Ghahremani** <tanaz@smartory.co>
- **Mohammad Hoseini** <mohammad@smartory.co>
- **Mohammad Javad Ahmadi** <mjahmadi@smartory.co>
- **Pouria Mehrabi** <pouria@smartory.co>
- **Amirhossein Nikoofard** <nikoofard@smartory.co>

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

## Pre-Print Paper: 
**When using this library in your research, we will be happy if you cite us! (or at least bring us some self-made pizza or burgers)** 
1. ‌
    ```bibtex
    @misc{ghahremani2024augmentory,
        title={AugmenTory: A Fast and Flexible Polygon Augmentation Library},
        author={Tanaz Ghahremani and Mohammad Hoseyni and Mohammad Javad Ahmadi and Pouria Mehrabi and Amirhossein Nikoofard},
        year={2024},
        eprint={2405.04442},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    ```
2. ‌
   ```plaintext
        Ghahremani, T., & Hoseyni, M., & Ahmadi, M. J., & Mehrabi, P., & Nikoofard, A. (2024). AugmenTory: A Fast and Flexible Polygon Augmentation Tool. arXiv preprint [arXiv:2405.04442](https://arxiv.org/abs/2405.04442).
   ```
