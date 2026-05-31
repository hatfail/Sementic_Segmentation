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
    
    import time
    keyframes = {}
    all_times = []
    key_indices = [0, total_frames // 2, total_frames - 1] if total_frames > 2 else [0]
    for idx in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        img = pil_img.resize(input_size, Image.BILINEAR)
        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(img_tensor, mean=mean, std=std)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            output = model(img_tensor)['out'][0]
            preds = torch.argmax(output, dim=0).cpu().numpy()
        end_time = time.time()
        all_times.append(end_time - start_time)

        mask_rgb = decode_segmentation_masks(preds, colormap, config['model']['num_classes'])

        if config['visualization']['restore_original_size']:
            mask_img = Image.fromarray(mask_rgb)
            mask_img = mask_img.resize((width, height), Image.NEAREST)
            mask_rgb = np.array(mask_img)

        overlayed = overlay_mask(pil_img, mask_rgb, alpha=config['visualization']['overlay_alpha'])

        # 保存关键帧
        if idx in key_indices:
            key_name = f"keyframe_{'start' if idx==0 else 'mid' if idx==total_frames//2 else 'end'}.png"
            Image.fromarray(overlayed).save(key_name)
            keyframes[idx] = key_name

        res_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
        out.write(res_bgr)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

    # FPS与单帧推理时间统计
    if all_times:
        avg_time = sum(all_times) / len(all_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"平均单帧推理时间: {avg_time*1000:.2f} ms, 平均FPS: {fps:.2f}")
    else:
        print("未统计到推理时间。")

    # 输出关键帧信息
    print("关键帧分割结果已保存：")
    for idx, fname in keyframes.items():
        print(f"  帧 {idx}: {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video or directory")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (single file mode)")
    args = parser.parse_args()

    # 判断是单文件还是目录
    if os.path.isdir(args.video):
        exts = ('.mp4', '.avi', '.mov', '.mkv')
        vid_list = [os.path.join(args.video, f) for f in os.listdir(args.video) if f.lower().endswith(exts)]
        vid_list.sort()
        print(f"检测到目录，批量推理 {len(vid_list)} 个视频...")
        for vid_path in vid_list:
            out_name = os.path.splitext(os.path.basename(vid_path))[0] + "_seg.mp4"
            infer_video(vid_path, args.config, out_name)
    else:
        out_path = args.output if args.output else os.path.splitext(os.path.basename(args.video))[0] + "_seg.mp4"
        infer_video(args.video, args.config, out_path)
