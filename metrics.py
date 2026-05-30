import torch
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, preds, labels):
        # preds: [B, H, W] or [B, C, H, W]
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)
            
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        mask = (labels >= 0) & (labels < self.num_classes) & (labels != self.ignore_index)
        hist = np.bincount(
            self.num_classes * labels[mask].astype(int) + preds[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist

    def get_results(self):
        hist = self.confusion_matrix
        
        # Pixel Accuracy
        pa = np.diag(hist).sum() / (hist.sum() + 1e-6)
        
        # mIoU
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-6)
        miou = np.nanmean(iou)
        
        # Dice
        # Dice = 2*TP / (2*TP + FP + FN)
        dice = (2 * np.diag(hist)) / (hist.sum(axis=1) + hist.sum(axis=0) + 1e-6)
        mdice = np.nanmean(dice)
        
        return {
            'Pixel Accuracy': pa,
            'mIoU': miou,
            'mDice': mdice
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
