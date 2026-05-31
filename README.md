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

### 小贴士（显存不足）
- 若出现 `CUDA Out of Memory`，在 `train.py` 中把 `batch_size` 从 4 调小为 2 或 1。
- 如果运行在有 GPU 但驱动不兼容的服务器，建议安装与驱动匹配的 PyTorch 版本，或请管理员更新驱动。

## 推断
### 单张图像
```
python inference_image.py --image path/to/image.jpg --output output.png
```
会加载 `config.yaml` 指定的 checkpoint（或使用预训练权重），生成带覆盖的可视化结果。

### 视频
```
python inference_video.py --video path/to/input.mp4 --output path/to/output.mp4
```
逐帧分割并输出合成的视频。

## 性能参考
- 在本项目配置下（`deeplabv3_resnet50`, input 320x320）：
  - GPU 训练得到的 best mIoU：约 **73.86%**
  - 在 CPU 上训练得到的 best mIoU（可作为对比）：约 **74.98%**

这两个数值在数据量与训练轮数受限（10 epochs、未使用大规模数据增强或额外数据集）的情况下属于合理范围。

## 常见问题
- 日志里显示 `CUDA initialization: The NVIDIA driver on your system is too old`：表示 PyTorch 与驱动版本不兼容，请安装与服务器驱动匹配的 PyTorch 或升级驱动。
- 若想提高 mIoU，建议：使用更高分辨率输入、数据增强（随机缩放/裁剪/翻转）、更多训练 epoch、使用更大训练集（如 SBD）以及多尺度测试/CRF 后处理。

## 复现实验与报告建议
在写报告时，建议包括：
- 数据划分方式（7:1:2）与数量
- 训练超参（学习率、优化器、batch size、epoch）
- 评估指标（PA、mIoU、Dice）与最终数值
- 复现实验中可能导致差异的因素（训练数据、增强、测试时间策略、硬件差异）

## 联系
如需我帮您：添加训练脚本的 Slurm/SGE 提交模板、加入数据增强、或写实验报告模板，请告诉我具体需求。