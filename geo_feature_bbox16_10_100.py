##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################
"""
Preprocess the absolute bounding box coordinates from --input_box_dir,
To convert these into relative coordinates, this script loads the corresponding images from coco/val2014/ and coco/train2014/, to get (img_width, img_height)

Input:
input_box_dir="/mydisk/Data/captioning_data/cocobu_adaptive_box"
info_filepath="/mydisk/Data/captioning_data/dataset_coco.json"
img_dir      ="/mydisk/Data/captioning_data/coco"

Output:
A directory containing all the boxes relative coordinates, as npy files.
"""

import os
import os.path as pth
from glob import glob
import json
import requests
from io import BytesIO
import numpy as np
import PIL.Image
import argparse


def _pil_to_nparray(pim):
    image = pim.convert("RGB")
    imageArray = np.array(image)
    return imageArray


def get_numpy_image(url_or_filepath):
    """
    Converts an image URL or filepath to its numpy array.

    :param str url_or_filepath:
        The URL or filepath of the image we want to convert
    :returns np.array:
        'RGB' np.array representing the image
    """
    if url_or_filepath.startswith(('http', 'www')):
        response = requests.get(url_or_filepath)
        pim = PIL.Image.open(BytesIO(response.content))
    else:
        pim = PIL.Image.open(url_or_filepath)

    return _pil_to_nparray(pim)


def get_bbox_relative_coords(params):
    input_box_dir = params['input_box_dir']
    info_filepath = params['input_json']
    img_dir = params['image_root']
    output_dir = params['output_dir']

    print("Reading coco dataset info")
    with open(info_filepath, "r") as infile:
        coco_dict = json.load(infile)

    coco_ids_to_paths = {str(img['cocoid']): pth.join(img_dir, img['filepath'], img['filename']) for img in coco_dict['images']}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    if not os.path.exists(img_dir):
        print(f"Directory does not exist: {input_box_dir}")

    box_files = sorted(glob(pth.join(input_box_dir, '*')))

    for ind, box_file in enumerate(box_files):
        if ind % 1000 == 0:
            print('Processed %d images (of %d)' % (ind, len(box_files)))

        filenumber = pth.splitext(pth.basename(box_file))[0]
        img_path = coco_ids_to_paths.get(filenumber)

        if not img_path:
            print(f"Image path for {filenumber} not found.")
            continue

        img_array = get_numpy_image(img_path)
        print(img_array.shape)

        height, width = img_array.shape[:2]
        print(img_array.shape[:2])

        box = np.load(box_file)
        x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

        # Width and Height of the Bounding Box
        widthb = x2 - x1  # box2 - box0
        heightb = y2 - y1  # box3 - box1

        # Area of the Bounding Box
        area = widthb * heightb

        # Aspect Ratio
        aspect_ratio = widthb / heightb

        # Center Coordinates
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Perimeter
        perimeter = 2 * (widthb + heightb)

        # Diagonal Length
        diagonal = np.sqrt(widthb ** 2 + heightb ** 2)

        # Margins from Image Edges
        left_margin = x1
        top_margin = y1
        right_margin = width - x2
        bottom_margin = height - y2

        # Distance to Image Center
        image_center_x = width / 2
        image_center_y = height / 2
        distance_to_center = np.sqrt((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2)

        # Normalize Features Relative to Image Dimensions
        relative_x1 = x1 / width
        relative_y1 = y1 / height
        relative_x2 = x2 / width
        relative_y2 = y2 / height
        relative_widthb = widthb / width
        relative_heightb = heightb / height
        relative_area = area / (width * height)
        relative_center_x = center_x / width
        relative_center_y = center_y / height
        relative_perimeter = perimeter / (2 * (width + height))
        relative_diagonal = diagonal / np.sqrt(width ** 2 + height ** 2)
        relative_left_margin = left_margin / width
        relative_top_margin = top_margin / height
        relative_right_margin = right_margin / width
        relative_bottom_margin = bottom_margin / height
        relative_distance_to_center = distance_to_center / np.sqrt(width ** 2 + height ** 2)

        # Aggregate Important Relative Features
        relative_features = np.column_stack(
            (relative_x1, relative_y1, relative_x2, relative_y2, relative_widthb, relative_heightb
             , relative_area, aspect_ratio, relative_center_x,
             relative_center_y, relative_perimeter, relative_diagonal,
             relative_left_margin, relative_top_margin, relative_right_margin,
             relative_bottom_margin, relative_distance_to_center))

        # Clip values to [0, 1] to ensure they are within valid range (except aspect_ratio)
        relative_features[:, :-1] = np.clip(relative_features[:, :-1], 0.0, 1.0)

        # Save the relative features
        new_filename = pth.join(output_dir, filenumber + '.npy')
        np.save(new_filename, relative_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default='D:\\Top-Down\\data\\dataset_coco.json',
                        help='Input JSON file to process into hdf5')
    parser.add_argument('--image_root', type=str, default='D:\\Top-Down\\data\\$IMAGE_ROOT',
                        help='Root path to prepend to image folder paths')
    parser.add_argument('--input_box_dir', type=str, default='F:\Feature_TOP_DOWN_10_100/up_down_100_box',
                        help='Directory containing the boxes of att feats')
    parser.add_argument('--output_dir', type=str, default='F:\Feature_TOP_DOWN_10_100/geometricL_features16_10_100',
                        help='Directory to save the relative coordinates of the bboxes in --input_box_dir')

    args = parser.parse_args()
    params = vars(args)  # Convert to ordinary dict
    get_bbox_relative_coords(params)
