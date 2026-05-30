import os
import argparse
import yaml
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

from visualizer import get_voc_colormap, decode_segmentation_masks, overlay_mask
from model_loader import load_trained_model

def infer_video(video_path, config_path, output_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['runtime']['device'] if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(config, device)
    
    colormap = get_voc_colormap()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video stream or file: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    input_size = config['preprocess']['input_size']
    mean = config['preprocess']['mean']
    std = config['preprocess']['std']
    
    for _ in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Preprocess
        img = pil_img.resize(input_size, Image.BILINEAR)
        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=mean, std=std)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(img_tensor)['out'][0]
            preds = torch.argmax(output, dim=0).cpu().numpy()
            
        # Visualizer
        mask_rgb = decode_segmentation_masks(preds, colormap, config['model']['num_classes'])
        
        if config['visualization']['restore_original_size']:
            mask_img = Image.fromarray(mask_rgb)
            mask_img = mask_img.resize((width, height), Image.NEAREST)
            mask_rgb = np.array(mask_img)
            
        overlayed = overlay_mask(pil_img, mask_rgb, alpha=config['visualization']['overlay_alpha'])
        
        # Convert back to BGR for cv2
        res_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
        out.write(res_bgr)
        
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video")
    args = parser.parse_args()
    
    infer_video(args.video, args.config, args.output)
