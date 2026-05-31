'''
Author: hatfail 1833943280@qq.com
Date: 2026-05-30 23:46:24
LastEditors: hatfail 1833943280@qq.com
LastEditTime: 2026-05-31 21:04:35
FilePath: \Project\visualizer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import cv2
from PIL import Image

def get_voc_colormap():
    # Returns 21 colors for PASCAL VOC
    return np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128]
    ])

def decode_segmentation_masks(mask, colormap, num_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, num_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def overlay_mask(image, mask_rgb, alpha=0.5):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # image and mask_rgb should be the same size
    if mask_rgb.shape[:2] != image.shape[:2]:
        mask_rgb = cv2.resize(mask_rgb, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    return overlay
