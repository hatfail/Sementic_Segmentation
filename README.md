<!--
 * @Author: hatfail 1833943280@qq.com
 * @Date: 2026-05-31 21:16:14
 * @LastEditors: hatfail 1833943280@qq.com
 * @LastEditTime: 2026-05-31 21:25:52
 * @FilePath: \Project\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# DeepLabV3 (ResNet50) PASCAL VOC 2012 复现项目

## 项目概述
本项目复现了基于 `deeplabv3_resnet50` 的语义分割在 PASCAL VOC2012 数据集上的训练与推断。
目标：训练模型并用自采图片/视频进行推断，评估指标包括 Pixel Accuracy、mIoU、Dice。

## 仓库结构
- `config.yaml`：训练与推断的配置文件（模型、预处理、数据路径、可视化设置等）。
- `dataset.py`：VOC 数据集加载与 7:1:2 划分（train/val/test）。
- `train.py`：训练主流程，保存 `checkpoints/best_deeplabv3_voc.pth`。
- `model_loader.py`：加载训练好模型的独立工具函数 `load_trained_model`。
- `inference_image.py`：对单张图片进行语义分割并保存/展示结果。
- `inference_video.py`：对视频帧逐帧分割并输出带蒙版覆盖的视频。
- `visualizer.py`：调色板与可视化工具（遮罩 overlay）。
- `metrics.py`：计算 Pixel Accuracy / mIoU / Dice 的工具。
- `requirements.txt`：依赖列表。

## 环境依赖
建议使用 Conda 创建虚拟环境并安装依赖：

```bash
python -m pip install -r requirements.txt
```

如果需要 GPU 支持，请安装与服务器驱动匹配的 PyTorch（示例：CUDA 11.8）：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

检查 GPU 可用性：

```bash
python -c "import torch; print('GPU可用:', torch.cuda.is_available())"
```

## 数据集准备
下载并解压 PASCAL VOC2012 数据集到项目目录下，使得路径为：

```
VOCdevkit/
  VOC2012/
    JPEGImages/
    SegmentationClass/
    ImageSets/
```

在 `config.yaml` 中默认 `data_root: VOCdevkit/VOC2012`，若放置位置不同请修改该字段。

## 配置说明
编辑 `config.yaml` 来修改：输入尺寸、均值/方差、类别数、checkpoint 路径、设备等。例如：

```yaml
model:
  name: deeplabv3_resnet50
  num_classes: 21
  checkpoint: checkpoints/best_deeplabv3_voc.pth

preprocess:
  input_size: [320, 320]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

data:
  data_root: VOCdevkit/VOC2012
  ignore_index: 255

visualization:
  overlay_alpha: 0.5
  restore_original_size: true

runtime:
  device: cuda
```

## 训练
在确认数据与环境准备就绪后运行：

```bash
python train.py
```

训练过程会打印进度并在验证集 mIoU 提升时保存最优模型到 `checkpoints/best_deeplabv3_voc.pth`。

## 推断
### 单张图像
```
python inference_image.py --image path/to/image.jpg --output output.png
python inference_image.py --image path/to/image_folder
```
会加载 `config.yaml` 指定的 checkpoint（或使用预训练权重），生成带覆盖的可视化结果。

### 视频
```
python inference_video.py --video path/to/input.mp4 --output path/to/output.mp4
python inference_video.py --video path/to/video_folder
```
逐帧分割并输出合成的视频。
