# MNIST 手写数字识别

基于 PyTorch 实现的 CNN 手写数字识别项目，提供训练、推理、评测的完整流程，支持 TensorBoard 可视化。

## 环境准备

### 依赖安装

```bash
pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --break-system-packages tensorboard matplotlib
```

如果机器有 NVIDIA GPU，将第一行替换为：

```bash
pip install --break-system-packages torch torchvision
```

### 验证安装

```bash
python3 -c "import torch; import torchvision; import tensorboard; import matplotlib; print('OK')"
```

## 项目结构

```
当前工作区/
├── config.py              # 所有可调参数集中管理
├── train.py               # 训练脚本（数据加载、训练循环、验证、保存模型）
├── predict.py             # 单张图片推理脚本（输出预测数字和置信度）
├── evaluate.py            # 批量评测脚本（遍历评测目录、生成报告和可视化）
├── models/
│   └── net.py             # CNN 神经网络模型定义
├── data/
│   ├── MNIST/             # MNIST 原始数据集（已下载，无需重新下载）
│   └── eval_images/       # 评测图片存放目录（你放入自己的手写数字图片）
├── logs/
│   └── mnist_experiment/  # TensorBoard 事件文件
└── results/               # 推理和评测的输出结果
    ├── evaluation_results.txt           # 评测文本报告
    ├── evaluation_visualization.png     # 评测结果可视化大图
    └── predict_<文件名>.png             # 单张推理可视化图
```

## 一、训练模型

### 基本训练

```bash
python3 train.py
```

训练过程中终端会实时输出每轮的 loss 和准确率：

```
Epoch [1/10] Train Loss: 0.1448 | Train Acc: 95.62% | Val Loss: 0.0613 | Val Acc: 98.16%
  -> Best model saved (Val Acc: 98.16%)
...
Training finished. Best Val Accuracy: 99.12%
```

训练完成后，最佳模型保存在 `models/mnist_model.pth`。

### 数据集说明

- **训练集**：MNIST 的 60000 张训练图片，其中 85% 用于训练、15% 用于验证
- **测试集**：MNIST 自带的 10000 张测试图片（评测脚本默认不直接用它，而是用你在 `data/eval_images/` 中放入的自定义图片）

数据会在首次运行时自动下载到 `data/MNIST/` 目录（当前已预置，不会重复下载）。

### 调整训练参数

修改 `config.py` 中的参数即可：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 64 | 每批训练的样本数 |
| `LEARNING_RATE` | 0.001 | 初始学习率（训练中会自动衰减） |
| `EPOCHS` | 10 | 训练轮数 |
| `NUM_WORKERS` | 2 | 数据加载的并行线程数 |

## 二、查看训练曲线（TensorBoard）

训练完成后，启动 TensorBoard 查看 loss 和 accuracy 曲线：

```bash
 tensorboard --logdir=Day05/MNIST_project/logs  
```

终端会输出类似 `http://localhost:6006` 的地址，在浏览器中打开即可看到：

- `Loss/Train` — 每个 epoch 的训练损失
- `Loss/Val` — 每个 epoch 的验证损失
- `Accuracy/Train` — 每个 epoch 的训练准确率
- `Accuracy/Val` — 每个 epoch 的验证准确率
- `Train/BatchLoss` — 每个 batch 的损失（可观察训练是否平稳）

## 三、单张图片推理

用于快速测试某张手写数字图片的识别结果。

```bash
python3 predict.py <你的图片路径>
```

### 示例

```bash
python3 predict.py data/eval_images/7_0000.png
```

输出：

```
Predicted digit: 7 (confidence: 100.00%)
Visualization saved to results/predict_7_0000.png
```

### 支持的图片格式

png、jpg、jpeg、bmp、tiff、webp。彩色图片会自动转为灰度，任意尺寸会缩放到 28x28。

### 输出说明

- **终端**：打印预测的数字和置信度
- **图片文件**：保存在 `results/predict_<原文件名>.png`，左侧是原始图片，右侧是预测结果

## 四、批量评测

### 步骤 1：准备评测图片

将你的手写数字图片放入 `data/eval_images/` 目录。

**图片命名建议**（用于自动统计准确率）：使用 `数字_编号.png` 的格式，例如：

```
data/eval_images/
├── 0_0001.png    # 手写数字 0
├── 0_0002.png    # 手写数字 0
├── 1_0001.png    # 手写数字 1
├── 1_0002.png    # 手写数字 1
├── 2_0001.png    # 手写数字 2
...
```

这样评测脚本会自动从文件名中提取真实标签并计算准确率。如果不想按此命名，也完全不影响推理——只是无法自动统计准确率。

支持格式：png、jpg、jpeg、bmp、tiff、webp。

### 步骤 2：运行评测

```bash
python3 evaluate.py
```

### 步骤 3：查看结果

评测完成后生成两类输出：

#### 文本报告 `results/evaluation_results.txt`

每张图片的详细预测，包含每个数字 0-9 的概率分布（可视化柱状图）：

```
============================================================
MNIST Handwritten Digit Recognition - Evaluation Results
============================================================

[001] 0_0003.png
     Predicted: 0
     Confidence: 1.0000 (100.00%)
     All Probabilities:
       0: ################################################# 1.0000
       1:  0.0000
       2:  0.0000
       ...
------------------------------------------------------------
Total images: 100
Correct predictions: 99
Accuracy (filename-based label): 99.00%
------------------------------------------------------------
```

#### 可视化图片 `results/evaluation_visualization.png`

将所有评测图片按网格排列，每张图上方标注预测数字和置信度，方便快速浏览整体效果。

## 五、模型结构

```
输入 (28x28 灰度图)
  └─ Conv2d(1→32, 3x3) + BatchNorm + ReLU + MaxPool(2x2)  → 14x14x32
  └─ Conv2d(32→64, 3x3) + BatchNorm + ReLU + MaxPool(2x2) → 7x7x64
  └─ Conv2d(64→128, 3x3) + BatchNorm + ReLU                → 7x7x128
  └─ Flatten                                                → 6272
  └─ Dropout(0.25)
  └─ Linear(6272→256) + ReLU + Dropout(0.5)
  └─ Linear(256→10)
输出: 10 类（数字 0-9 的 logits）
```

## 六、完整工作流程总结

如果你是第一次使用这套代码，按以下流程操作：

```
1. 安装依赖（见"环境准备"）
       ↓
2. python3 train.py              → 训练模型
       ↓
3. tensorboard --logdir=logs/    → 查看训练曲线，确认模型收敛
       ↓
4. 准备自己的手写数字图片，放入 data/eval_images/
       ↓
5. python3 predict.py <图片路径>  → 单张测试，验证模型效果
       ↓
6. python3 evaluate.py            → 批量评测，生成完整报告
       ↓
7. 查看 results/evaluation_results.txt 和 evaluation_visualization.png
```

## 常见问题

**Q: 训练时需要联网吗？**
A: 不需要。MNIST 数据集已预置在 `data/MNIST/` 中。

**Q: 自己的图片尺寸不一致怎么办？**
A: 不需要手动调整。推理和评测代码会自动将任意尺寸图片缩放为 28x28 灰度图。

**Q: 评测结果准确率怎么算的？**
A: 脚本从文件名中提取第一个下划线前的数字作为真实标签（如 `3_hello.png` → 标签为 3），与模型预测对比。如果文件名不符合此规则，会跳过准确率计算但不影响推理结果。

**Q: 如何提高准确率？**
A: 可以在 `config.py` 中增加 `EPOCHS`、调整 `LEARNING_RATE`，或在 `models/net.py` 中修改网络结构（增加层数、通道数等）。

**Q: 没有显示器怎么看可视化图？**
A: 推理和评测脚本在检测不到图形界面时会自动将可视化保存为 PNG 文件到 `results/` 目录，不会报错。
