import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import VOCSegmentationDataset
from metrics import SegmentationMetrics
import torchvision.models.segmentation as segmentation
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

def parse_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model(config):
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    
    if model_name == 'deeplabv3_resnet50':
        # Load pretrained model
        model = segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        # Modify the classifier to match the number of classes (21 for VOC)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError(f"Model {model_name} is not supported.")
        
    return model

def train(config_path="config.yaml"):
    config = parse_config(config_path)
    device = torch.device(config['runtime']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    train_dataset = VOCSegmentationDataset(config['data']['data_root'], split='train', config=config)
    val_dataset = VOCSegmentationDataset(config['data']['data_root'], split='val', config=config)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Model Setup
    model = get_model(config).to(device)
    
    # Loss and Optimizer
    ignore_index = config['data']['ignore_index']
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    # Metrics
    metrics = SegmentationMetrics(num_classes=config['model']['num_classes'], ignore_index=ignore_index)
    
    num_epochs = 10
    best_miou = 0.0
    checkpoint_dir = os.path.dirname(config['model']['checkpoint'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")
        
        # Validation
        model.eval()
        metrics.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in loop:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                metrics.update(outputs, masks)
                
        results = metrics.get_results()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} | PA: {results['Pixel Accuracy']:.4f} | mIoU: {results['mIoU']:.4f} | mDice: {results['mDice']:.4f}")
        
        if results['mIoU'] > best_miou:
            best_miou = results['mIoU']
            torch.save(model.state_dict(), config['model']['checkpoint'])
            print(f"Saved best model with mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    train()

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import VOCSegmentationDataset
from metrics import SegmentationMetrics
import torchvision.models.segmentation as segmentation
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

def parse_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model(config):
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    
    if model_name == 'deeplabv3_resnet50':
        # Load pretrained model
        model = segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        # Modify the classifier to match the number of classes (21 for VOC)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError(f"Model {model_name} is not supported.")
        
    return model

def train(config_path="config.yaml"):
    config = parse_config(config_path)
    device = torch.device(config['runtime']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    train_dataset = VOCSegmentationDataset(config['data']['data_root'], split='train', config=config)
    val_dataset = VOCSegmentationDataset(config['data']['data_root'], split='val', config=config)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Model Setup
    model = get_model(config).to(device)
    
    # Loss and Optimizer
    ignore_index = config['data']['ignore_index']
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    # Metrics
    metrics = SegmentationMetrics(num_classes=config['model']['num_classes'], ignore_index=ignore_index)
    
    num_epochs = 10
    best_miou = 0.0
    checkpoint_dir = os.path.dirname(config['model']['checkpoint'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f}")
        
        # Validation
        model.eval()
        metrics.reset()
        val_loss = 0.0
        
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in loop:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                metrics.update(outputs, masks)
                
        results = metrics.get_results()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} | PA: {results['Pixel Accuracy']:.4f} | mIoU: {results['mIoU']:.4f} | mDice: {results['mDice']:.4f}")
        
        if results['mIoU'] > best_miou:
            best_miou = results['mIoU']
            torch.save(model.state_dict(), config['model']['checkpoint'])
            print(f"Saved best model with mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    train()
