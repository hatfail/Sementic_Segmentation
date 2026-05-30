import os
import argparse
import yaml
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt

from visualizer import get_voc_colormap, decode_segmentation_masks, overlay_mask
from model_loader import load_trained_model

def infer_image(image_path, config_path, output_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['runtime']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_trained_model(config, device)
    
    # Prepare image
    orig_img = Image.open(image_path).convert('RGB')
    orig_size = orig_img.size
    
    img = orig_img.resize(config['preprocess']['input_size'], Image.BILINEAR)
    img_tensor = TF.to_tensor(img)
    img_tensor = TF.normalize(img_tensor, mean=config['preprocess']['mean'], std=config['preprocess']['std'])
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)['out'][0]
        preds = torch.argmax(output, dim=0).cpu().numpy()
        
    colormap = get_voc_colormap()
    mask_rgb = decode_segmentation_masks(preds, colormap, config['model']['num_classes'])
    
    if config['visualization']['restore_original_size']:
        mask_img = Image.fromarray(mask_rgb)
        mask_img = mask_img.resize(orig_size, Image.NEAREST)
        mask_rgb = np.array(mask_img)
        
    overlayed = overlay_mask(orig_img, mask_rgb, alpha=config['visualization']['overlay_alpha'])
    
    if output_path:
        Image.fromarray(overlayed).save(output_path)
        print(f"Saved inference result to {output_path}")
    else:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(overlayed)
        plt.title('Segmentation Prediction')
        plt.axis('off')
        
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image")
    args = parser.parse_args()
    
    infer_image(args.image, args.config, args.output)
