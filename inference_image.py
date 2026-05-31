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

    # 文件名自动编号
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # 查找当前目录下已有的ImageXX_Original_Image等，自动编号
    import glob
    exist_imgs = glob.glob("Image*_Original_Image.*")
    exist_ids = [int(f.split('_')[0][5:]) for f in exist_imgs if f.split('_')[0][5:].isdigit()]
    idx = max(exist_ids) + 1 if exist_ids else 1

    orig_out = f"Image{idx:02d}_Original_Image.png"
    mask_out = f"Image{idx:02d}_Predicted_Mask.png"
    overlay_out = f"Image{idx:02d}_Overlay_Result.png"

    # 保存三张图片
    orig_img.save(orig_out)
    Image.fromarray(mask_rgb).save(mask_out)
    Image.fromarray(overlayed).save(overlay_out)
    print(f"已保存: {orig_out}, {mask_out}, {overlay_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Path to save output image (single file mode)")
    args = parser.parse_args()

    # 判断是单文件还是目录
    if os.path.isdir(args.image):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        img_list = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.lower().endswith(exts)]
        img_list.sort()
        print(f"检测到目录，批量推理 {len(img_list)} 张图片...")
        for img_path in img_list:
            infer_image(img_path, args.config, None)
    else:
        infer_image(args.image, args.config, args.output)
