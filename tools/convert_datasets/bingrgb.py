# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from PIL import Image
import numpy as np
import shutil
import argparse

# Class map for BingRGB dataset
class_map = {
    (0, 0, 0): 0,            # background
    (0, 255, 0): 1,          # farmland
    (0, 0, 255): 2,          # water
    (0, 255, 255): 3,        # forest
    (128, 0, 0): 4,          # urban_structure
    (255, 0, 255): 5,        # rural_built_up
    (255, 0, 0): 6,          # urban_built_up
    (160, 160, 164): 7,      # road
    (255, 255, 0): 8,        # meadow
    (255, 251, 240): 9,      # marshland
    (128, 0, 128): 10        # brick_factory
}

def encode_bw_mask(rgb_mask, class_map=class_map):
    rgb_mask_array = np.array(rgb_mask)
    bw_mask = np.zeros((rgb_mask_array.shape[0], rgb_mask_array.shape[1]), dtype=np.uint8)
    for rgb_val, class_label in class_map.items():
        indices = np.where(np.all(rgb_mask_array == rgb_val, axis=-1))
        bw_mask[indices] = class_label
    return Image.fromarray(bw_mask)

def convert_bingrgb_to_trainID(image_dir, label_dir, out_img_dir, out_label_dir, is_train=True):
    img_subdir = 'train' if is_train else 'test'
    images = os.listdir(osp.join(image_dir))
    labels = os.listdir(osp.join(label_dir))
    
    os.makedirs(osp.join(out_img_dir, img_subdir), exist_ok=True)
    os.makedirs(osp.join(out_label_dir, img_subdir), exist_ok=True)
    
    for img_file, label_file in zip(images, labels):
        # Copy image file to output folder
        shutil.copy(osp.join(image_dir, 'image_patches', img_file), osp.join(out_img_dir, img_subdir, img_file))
        
        # Convert and save label file
        rgb_mask = Image.open(osp.join(label_dir, 'label_patches', label_file))
        bw_mask = encode_bw_mask(rgb_mask)
        bw_mask.save(osp.join(out_label_dir, img_subdir, label_file.replace('.png', '_labelTrainIds.png')))

def main():
    parser = argparse.ArgumentParser(description='Convert BingRGB LULC dataset to segmentation format.')
    parser.add_argument('data_path', help='Path to BingRGB dataset')
    parser.add_argument('-o', '--out_dir', help='Output path for converted dataset', default='output')
    args = parser.parse_args()

    data_path = args.data_path
    out_dir = args.out_dir

    train_img_dir = osp.join(data_path, 'train', 'image_patches')
    train_label_dir = osp.join(data_path, 'train', 'label_patches')
    test_img_dir = osp.join(data_path, 'test', 'image_patches')
    test_label_dir = osp.join(data_path, 'test', 'label_patches')

    convert_bingrgb_to_trainID(train_img_dir, train_label_dir, osp.join(out_dir, 'images'), osp.join(out_dir, 'annotations'), is_train=True)
    convert_bingrgb_to_trainID(test_img_dir, test_label_dir, osp.join(out_dir, 'images'), osp.join(out_dir, 'annotations'), is_train=False)

    print('Conversion completed!')

if __name__ == '__main__':
    main()
