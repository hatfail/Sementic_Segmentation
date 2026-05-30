import os
import torch
from train import get_model

def load_trained_model(config, device):
    """
    根据配置文件加载训练好的模型，供推理（图像/视频）使用。
    """
    model = get_model(config)
    checkpoint_path = config['model']['checkpoint']
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained weights.")
        
    model.to(device)
    model.eval()  # 设置为评估模式
    
    return model
